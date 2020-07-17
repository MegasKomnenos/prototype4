#![feature(get_mut_unchecked)]

extern crate num_cpus;

use legion::prelude::*;
use legion::entity::Entity;
use legion::systems::SystemBuilder;
use legion::systems::resource::Resources;
use legion::systems::schedule::Schedule;
use rayon::ThreadPool;
use rayon::ThreadPoolBuilder;

use std::sync::Arc;
use std::sync::Barrier;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::Ordering;
use std::sync::mpsc::Sender;
use std::sync::mpsc::Receiver;
use std::sync::mpsc::channel;
use std::collections::HashMap;
use std::any::Any;

enum LoopEvent {
    RemoveEntity(Entity),
    ChangeComponent(Entity, Wrapper<Box<dyn Any>>, fn(&mut World, &Entity, Box<dyn Any>)),
    ChangeResource(Wrapper<Box<dyn Any>>, fn(&mut Resources, Box<dyn Any>)),
}

struct Wrapper<T> {
    item: T,
}

impl<T: Clone> Clone for Wrapper<T> {
    fn clone(&self) -> Self {
        Wrapper {
            item: self.item.clone()
        }
    }
}

unsafe impl<T> Send for Wrapper<T> {}
unsafe impl<T> Sync for Wrapper<T> {}

struct Loop {
    root: Arc<Barrier>,
    world: World,
    resources: Resources,
    schedule: Wrapper<Schedule>,
    events: Wrapper<Receiver<LoopEvent>>,
    on_schedule_start: Option<fn(&Resources, &Receiver<LoopEvent>)>,
    on_schedule_end: Option<fn(&Resources, &Receiver<LoopEvent>)>,
}

impl Loop {
    fn new(root: Arc<Barrier>, world: World, mut resources: Resources, schedule: Schedule) -> Self {
        let (producer, receiver) = channel::<LoopEvent>();

        resources.insert(Wrapper { item: producer });
        resources.insert(HashMap::<String, Wrapper<Sender<LoopEvent>>>::new());

        Loop {
            root,
            world,
            resources,
            schedule: Wrapper { item: schedule },
            events: Wrapper { item: receiver },
            on_schedule_start: None,
            on_schedule_end: None,
        }
    }

    fn start(&mut self, pool: &ThreadPool) {
        pool.install(|| {
            loop {
                if let Some(func) = self.on_schedule_start {
                    func(&self.resources, &self.events.item);
                }

                self.schedule.item.execute(&mut self.world, &mut self.resources);
                
                if let Some(func) = self.on_schedule_end {
                    func(&self.resources, &self.events.item);
                }
            }

            self.root.wait();
        });
    }

    fn handle_events(&mut self) {
        for event in self.events.item.try_iter() {
            match event {
                LoopEvent::RemoveEntity(entity) => {
                    self.world.delete(entity);
                },
                LoopEvent::ChangeComponent(entity, wrapper, func) => {
                    func(&mut self.world, &entity, wrapper.item);
                },
                LoopEvent::ChangeResource(wrapper, func) => {
                    func(&mut self.resources, wrapper.item);
                }
            }
        }
    }

    fn set_on_schedule_start(&mut self, func: fn(&Resources, &Receiver<LoopEvent>)) {
        self.on_schedule_start = Some(func);
    }
    fn set_on_schedule_end(&mut self, func: fn(&Resources, &Receiver<LoopEvent>)) {
        self.on_schedule_end = Some(func);
    }
}

struct Link {
    root: Arc<Barrier>,
    from: Arc<Loop>,
    to: Arc<Loop>,
    barrier: Arc<Barrier>,
    update: fn(Arc<Loop>, Arc<Loop>),
    on_update_start: Option<fn(Arc<Loop>, Arc<Loop>)>,
    on_update_end: Option<fn(Arc<Loop>, Arc<Loop>)>,
}

impl Link {
    fn new(root: Arc<Barrier>, loops: &Vec<Arc<Loop>>, names: &Vec<String>, from: String, to: String, update: fn(Arc<Loop>, Arc<Loop>)) -> Self {
        let mut from = loops[names.iter().position(|x| *x == from).unwrap()].clone();
        let mut to = loops[names.iter().position(|x| *x == to).unwrap()].clone();
        let barrier = Arc::new(Barrier::new(3));

        unsafe {
            Arc::get_mut_unchecked(&mut from).resources.insert(barrier.clone());
            Arc::get_mut_unchecked(&mut to).resources.insert(barrier.clone());
        }

        Link {
            root,
            from,
            to,
            barrier,
            update,
            on_update_start: None,
            on_update_end: None,
        }
    }

    fn start(&self, pool: &ThreadPool) {
        pool.install(|| {
            loop {
                self.barrier.wait();

                if let Some(func) = self.on_update_start {
                    func(self.from.clone(), self.to.clone());
                }

                (self.update)(self.from.clone(), self.to.clone());

                if let Some(func) = self.on_update_end {
                    func(self.from.clone(), self.to.clone());
                }
            }

            self.root.wait();
        });
    }

    fn set_on_update_start(&mut self, func: fn(Arc<Loop>, Arc<Loop>)) {
        self.on_update_start = Some(func);
    }
    fn set_on_update_end(&mut self, func: fn(Arc<Loop>, Arc<Loop>)) {
        self.on_update_end = Some(func);
    }
}

struct Core {
    root: Arc<Barrier>,
    universe: Universe,
    loops: Vec<Arc<Loop>>,
    links: Vec<Link>,
    pools: Vec<ThreadPool>,
    pools_link: Vec<ThreadPool>,
    names: Vec<String>,
}

impl Core {
    fn new() -> Self {
        let root = Arc::new(Barrier::new(4));
        let universe = Universe::new();

        Core {
            root,
            universe,
            loops: Vec::new(),
            links: Vec::new(),
            pools: Vec::new(),
            pools_link: Vec::new(),
            names: Vec::new(),
        }
    }

    fn load_application(&mut self) {
        self.names.push("Application".to_string());
        self.pools.push(ThreadPoolBuilder::new().num_threads(1).build().unwrap());
        self.loops.push(Arc::new(Loop::new(self.root.clone(), self.universe.create_world(), Resources::default(), Schedule::builder().build())));
    }

    fn load_systems(&mut self) {
        self.names.push("Systems".to_string());
        self.pools.push(ThreadPoolBuilder::new().num_threads(num_cpus::get() - 1).build().unwrap());
        self.loops.push(Arc::new(Loop::new(self.root.clone(), self.universe.create_world(), Resources::default(), Schedule::builder().build())));
    }

    fn load_event_channels(&mut self) {
        let i_app = self.names.iter().position(|x| *x == "Application".to_string()).unwrap();
        let i_sys = self.names.iter().position(|x| *x == "Systems".to_string()).unwrap();

        let mut loop_app = self.loops[i_app].clone();
        let mut loop_sys = self.loops[i_sys].clone();

        unsafe {
            let sender_sys = loop_sys.resources.get::<Wrapper<Sender<LoopEvent>>>().unwrap();
            Arc::get_mut_unchecked(&mut loop_app).resources.get_mut::<HashMap<String, Wrapper<Sender<LoopEvent>>>>().unwrap().insert(self.names[i_sys].clone(), sender_sys.clone());
        }
        unsafe {
            let sender_app = loop_app.resources.get::<Wrapper<Sender<LoopEvent>>>().unwrap();
            Arc::get_mut_unchecked(&mut loop_sys).resources.get_mut::<HashMap<String, Wrapper<Sender<LoopEvent>>>>().unwrap().insert(self.names[i_app].clone(), sender_app.clone());
        }
    }

    fn load_link(&mut self) {
        self.links.push(Link::new(self.root.clone(), &self.loops, &self.names, "Systems".to_string(), "Application".to_string(), |_, _| {}));
        self.pools_link.push(ThreadPoolBuilder::new().num_threads(num_cpus::get()).build().unwrap());
    }

    fn run(&mut self) {
        for (i, pool) in self.pools_link.iter().enumerate() {
            self.links[i].start(pool);
        }
        for (i, pool) in self.pools.iter().enumerate() {
            unsafe {
                Arc::get_mut_unchecked(&mut self.loops[i]).start(pool);
            }
        }

        self.root.wait();
    }
}

fn main() {
}

use legion::prelude::*;
use legion::entity::Entity;
use legion::storage::ComponentTypeId;
use legion::systems::SystemBuilder;
use legion::systems::resource::Resources;
use legion::systems::schedule::Schedule;
use rayon::ThreadPool;
use rayon::ThreadPoolBuilder;

use std::sync::mpsc::{ Sender, Receiver, channel };
use std::collections::HashMap;

enum LoopEvent {
    Enable,
    Disable,
    RemoveEntity(Entity),
    RemoveComponent(Entity, ComponentTypeId),
    ChangeComponent(Entity, ComponentTypeId, f32, fn(&mut World, &mut Resources, &Entity, &ComponentTypeId, f32)),
}

struct ScheduleWrapper {
    schedule: Schedule,
}

unsafe impl Send for ScheduleWrapper {}

#[derive(Clone)]
struct SenderWrapper {
    sender: Sender<LoopEvent>,
}

unsafe impl Send for SenderWrapper {}
unsafe impl Sync for SenderWrapper {}

struct Loop {
    name: String,
    enabled: bool,
    running: bool,
    waiting: u8,
    world: World,
    resources: Resources,
    schedule: ScheduleWrapper,
    receiver: Receiver<LoopEvent>,
}

impl Loop {
    fn new(name: String, world: World, mut resources: Resources, schedule: Schedule) -> Self {
        let (producer, receiver) = channel::<LoopEvent>();

        let sender = SenderWrapper { sender: producer };
        let mut senders = HashMap::new();
        senders.insert(name.clone(), sender.clone());

        resources.insert(sender);
        resources.insert(senders);

        Loop {
            name,
            enabled: false,
            running: false,
            waiting: 0,
            world,
            resources,
            schedule: ScheduleWrapper { schedule },
            receiver,
        }
    }

    fn execute(&mut self, pool: &mut ThreadPool) {
        if !self.enabled || self.waiting >= 1 {
            pool.install(|| {
                self.handle_event();
            });
        } else if !self.running {
            self.running = true;

            pool.install(|| {
                self.schedule.schedule.execute(&mut self.world, &mut self.resources);
                self.handle_event();
                self.running = false;
            });
        }
    }

    fn subscribe(&mut self, lp: &Loop) {
        let mut senders = self.resources.get_mut::<HashMap<String, SenderWrapper>>().unwrap();
        let sender = lp.resources.get::<SenderWrapper>().unwrap();
        senders.insert(self.name.clone(), sender.clone());
    }
    
    fn unsubscribe(&mut self, lp: &Loop) {
        let mut senders = self.resources.get_mut::<HashMap<String, SenderWrapper>>().unwrap();
        senders.remove(&lp.name);
    }

    fn handle_event(&mut self) {
        for event in self.receiver.try_iter() {
            match event {
                LoopEvent::Enable => self.enabled = true,
                LoopEvent::Disable => self.enabled = false,
                
                LoopEvent::RemoveEntity(entity) => {
                    self.world.delete(entity);
                },
                LoopEvent::RemoveComponent(entity, id) => {
                    self.world.move_entity(entity, &[], &[id], &[], &[]);
                }
                LoopEvent::ChangeComponent(entity, id, val, func) => {
                    func(&mut self.world, &mut self.resources, &entity, &id, val);
                }
            }
        }
    }
}

struct Link {
    from: String,
    to: String,
    from_id: Option<usize>,
    to_id: Option<usize>,
    waiting: bool,
    update: fn(&mut Vec<Loop>, usize, usize),
}

impl Link {
    fn new(from: String, to: String, update: fn(&mut Vec<Loop>, usize, usize)) -> Self {
        Link {
            from,
            to,
            from_id: None,
            to_id: None,
            waiting: false,
            update,
        }
    }
}

struct LoopManager {
    loops: Vec<Loop>,
    pools: Vec<ThreadPool>,
    links: Vec<Link>,
    name_id: HashMap<String, usize>,
}

impl LoopManager {
    fn new() -> Self {
        LoopManager {
            loops: Vec::new(),
            pools: Vec::new(),
            links: Vec::new(),
            name_id: HashMap::new(),
        }
    }

    fn add_loop(&mut self, lp: Loop, worker_num: usize) {
        self.name_id.insert(lp.name.clone(), self.name_id.len());
        self.loops.push(lp);
        self.pools.push(ThreadPoolBuilder::new().num_threads(worker_num).build().unwrap());
    }

    fn add_link(&mut self, mut link: Link) {
        link.from_id = Some(self.name_id.get(&link.from).unwrap().clone());
        link.to_id = Some(self.name_id.get(&link.to).unwrap().clone());
        self.links.push(link);
    }

    fn execute(&mut self) {
        loop {
            for link in self.links.iter_mut() {
                let from_id = link.from_id.unwrap();
                let to_id = link.to_id.unwrap();

                if self.loops[from_id].enabled && self.loops[to_id].enabled {
                    if !link.waiting {
                        if !self.loops[from_id].running {
                            link.waiting = true;
                            self.loops[from_id].waiting += 1;
                        }
                    }
                    if link.waiting && !self.loops[to_id].running {
                        link.waiting = false;
                        self.loops[from_id].waiting -= 1;

                        (link.update)(&mut self.loops, from_id, to_id);
                    }
                } else if link.waiting {
                    link.waiting = false;
                    self.loops[from_id].waiting -= 1;
                }
            }

            for (i, lp) in self.loops.iter_mut().enumerate() {
                lp.execute(&mut self.pools[i]);
            }
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct Test {
    x: f32,
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct Tag {
    id: usize,
    other: Option<Entity>,
}

fn main() {
    let universe = Universe::new();

    let mut world0 = universe.create_world();
    let mut world1 = universe.create_world();

    let num_of_countries = 100;

    world0.insert(
        (),
        (0..num_of_countries).map(|i| (Test { x: 0. }, Tag { id: i, other: None }))
    );
    world1.insert(
        (),
        (0..num_of_countries).map(|i| (Test { x: 0. }, Tag { id: i, other: None }))
    );

    let query = <Write<Tag>>::query();

    for (id_tag, mut tag) in query.iter_entities_mut(&mut world0) {
        for (id_other, mut other) in query.iter_entities_mut(&mut world1) {
            if tag.id == other.id {
                tag.other = Some(id_other.clone());
                other.other = Some(id_tag.clone());
            }
        }
    }

    let system0 = SystemBuilder::<()>::new("Test System 0")
            .with_query(<Write<Test>>::query())
            .build(move |_, world, _, queries| {
                for mut test in queries.iter_mut(&mut *world) {
                    test.x += 1.;
                }
            });
    let system1 = SystemBuilder::<()>::new("Test System 1")
            .with_query(<(Read<Test>, Read<Tag>)>::query())
            .build(move |_, world, _, queries| {
                for (test, tag) in queries.iter(world) {
                    println!("{} {}", test.x, tag.id);
                }
            });

    let schedule0 = Schedule::builder()
            .add_system(system0)
            .build();
    let schedule1 = Schedule::builder()
            .add_system(system1)
            .build();

    let loop0 = Loop::new("Test Loop 0".to_string(), world0, Default::default(), schedule0);
    let loop1 = Loop::new("Test Loop 1".to_string(), world1, Default::default(), schedule1);

    let link = Link::new("Test Loop 0".to_string(), "Test Loop 1".to_string(), |loops, from, to| {
        let mut storage = HashMap::<Entity, f32>::new();

        for (test, tag) in <(Read<Test>, Read<Tag>)>::query().iter(&loops[from].world) {
            storage.insert(tag.other.unwrap(), test.x);
        }
        for (entity, mut test) in <Write<Test>>::query().iter_entities_mut(&mut loops[to].world) {
            test.x = storage.get(&entity).unwrap() + 1.;
        }
    });

    let mut app = LoopManager::new();

    app.add_loop(loop0, 1);
    app.add_loop(loop1, 1);
    app.add_link(link);

    app.execute();
}

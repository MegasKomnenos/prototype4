#![feature(get_mut_unchecked)]
#![feature(cmp_min_max_by)]

extern crate num_cpus;

mod map;

use legion::prelude::*;
use legion::entity::Entity;
use legion::systems::SystemBuilder;
use legion::systems::resource::Resources;
use legion::systems::schedule::Schedule;
use rayon::ThreadPool;
use rayon::ThreadPoolBuilder;

use ron::ser::to_writer;

use rand::thread_rng;
use rand::Rng;

use std::sync::Arc;
use std::sync::Barrier;
use std::sync::Mutex;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::Ordering;
use std::sync::mpsc::Sender;
use std::sync::mpsc::Receiver;
use std::sync::mpsc::channel;
use std::collections::HashMap;
use std::any::Any;
use std::path::PathBuf;
use std::fs::File;
use std::cmp::max_by;
use std::cmp::min_by;

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

enum LoopEvent {
    RemoveEntity(Entity),
    ChangeComponent(Entity, Wrapper<Box<dyn Any>>, fn(&mut World, &Entity, Box<dyn Any>)),
    ChangeResource(Wrapper<Box<dyn Any>>, fn(&mut Resources, Box<dyn Any>)),
}

const SET: fn(&mut f32, &f32) = |x, y| *x = *y;
const MAX: fn(&mut f32, &f32) = |x, y| *x = max_by(*x, *y, |x, y| x.partial_cmp(y).unwrap());
const MIN: fn(&mut f32, &f32) = |x, y| *x = min_by(*x, *y, |x, y| x.partial_cmp(y).unwrap());
const ADD: fn(&mut f32, &f32) = |x, y| *x += y;
const SUBT: fn(&mut f32, &f32) = |x, y| *x -= y;
const MULT: fn(&mut f32, &f32) = |x, y| *x *= y;
const DIV: fn(&mut f32, &f32) = |x, y| *x /= y;
const POW: fn(&mut f32, &f32) = |x, y| *x = x.powf(*y);
const ROOT: fn(&mut f32, &f32) = |x, y| *x = x.powf(1./y);
const LOG: fn(&mut f32, &f32) = |x, y| *x = x.log(*y);

pub fn get_func(name: &String) -> fn(&mut f32, &f32) {
    match name.as_str() {
        "SET" => SET,
        "MAX" => MAX,
        "MIN" => MIN,
        "ADD" => ADD,
        "SUBT" => SUBT,
        "MULT" => MULT,
        "DIV" => DIV,
        "POW" => POW,
        "ROOT" => ROOT,
        "LOG" => LOG,
        _ => |_, _| (),
    }
}

struct Value {
    name: String,
    update: bool,
    base: f32,
    value: f32,
    funcs: Vec<fn(&mut f32, &f32)>,
    parents: Vec<usize>,
    children: Vec<usize>,
}

struct ValueHandle {
    value: f32,
    change: f32,
}

struct ValueManager {
    handles: Vec<Arc<ValueHandle>>,
    values: Vec<Value>,
    names: HashMap<String, usize>,
}

impl ValueManager {
    fn new() -> Self {
        let mut value = ValueManager {
            handles: Vec::new(),
            values: Vec::new(),
            names: HashMap::new(),
        };

        value.add_value("0".to_string(), 0., Vec::new(), Vec::new());
        value.add_value("1".to_string(), 1., Vec::new(), Vec::new());
        value.add_value("-1".to_string(), -1., Vec::new(), Vec::new());
        value.add_value("2".to_string(), 2., Vec::new(), Vec::new());
        value.add_value("-2".to_string(), -2., Vec::new(), Vec::new());

        return value;
    }

    fn add_value(&mut self, name: String, base: f32, funcs: Vec<String>, parents: Vec<String>) -> Arc<ValueHandle> {
        self.names.insert(name.clone(), self.names.len());
        let funcs: Vec<fn(&mut f32, &f32)> = funcs.iter().map(|n| get_func(n)).collect();
        let parents: Vec<usize> = parents.iter().map(|n| self.names.get(n).unwrap().clone()).collect();

        let mut value = base;

        for &parent in parents.iter() {
            self.values[parent].children.push(self.names.get(&name).unwrap().clone());
        }
        for (i, func) in funcs.iter().enumerate() {
            func(&mut value, &self.handles[parents[i]].value);
        }

        self.values.push(Value {
            name,
            update: false,
            base,
            value,
            funcs,
            parents,
            children: Vec::new(),
        });
        self.handles.push(Arc::new(ValueHandle { value, change: 0. }));

        return self.handles.last().unwrap().clone();
    }

    fn remove_value(&mut self, name: String) {
        let &i = self.names.get(&name).unwrap();

        for ii in (i..self.values.len()).rev() {
            *self.names.get_mut(&self.values[ii].name).unwrap() -= 1;

            for value in self.values.iter_mut() {
                for parent in value.parents.iter_mut() {
                    if *parent == ii {
                        *parent -= 1;
                    }
                }
                for child in value.children.iter_mut() {
                    if *child == ii {
                        *child -= 1;
                    }
                }
            }
        }

        self.handles.remove(i);
        self.values.remove(i);
        self.names.remove(&name);
    }

    fn update(&mut self) {
        let mut update = Vec::new();

        for (i, handle) in self.handles.iter_mut().enumerate() {
            let handle = unsafe { Arc::get_mut_unchecked(handle) };

            if handle.change != 0. {
                update.push(i);

                self.values[i].base += handle.change;
                handle.change = 0.;
            }
        }

        let mut stack = update.clone();

        while !stack.is_empty() {
            let i = stack.pop().unwrap();

            for child in self.values[i].children.iter() {
                if !update.contains(child) {
                    update.push(*child);
                    stack.push(*child);
                }
            }
        }

        while !update.is_empty() {
            let mut is = Vec::new();

            'outer: for ii in (0..update.len()).rev() {
                let i = update[ii].clone();

                for parent in self.values[i].parents.iter() {
                    if update.contains(parent) {
                        continue 'outer;
                    }
                }

                is.push(i);
                update.remove(ii);
            }

            for &i in is.iter() {
                let mut value = self.values[i].base;

                for (ii, func) in self.values[i].funcs.iter().enumerate() {
                    func(&mut value, &self.handles[self.values[i].parents[ii]].value);
                }

                self.values[i].value = value;
                unsafe { Arc::get_mut_unchecked(&mut self.handles[i]).value = value; }
            }
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct Pixel;
#[derive(Clone, Copy, Debug, PartialEq)]
struct Lake;
#[derive(Clone, Copy, Debug, PartialEq)]
struct Sea;

struct Owned { item: Entity }
struct Owns { item: Vec<Entity> }
struct Name { item: String }
struct Water { item: f64 }
struct River { item: f64 }
struct Rain { item: f64 }
struct Heat { item: f64 }
struct Height { item: f64 }
struct Veget { item: f64 }
struct Neighb { item: Vec<Entity> }


fn handle_event(world: &mut World, resources: &mut Resources, events: &Receiver<LoopEvent>) {
    for event in events.try_iter() {
        match event {
            LoopEvent::RemoveEntity(entity) => {
                world.delete(entity);
            },
            LoopEvent::ChangeComponent(entity, wrapper, func) => {
                func(world, &entity, wrapper.item);
            },
            LoopEvent::ChangeResource(wrapper, func) => {
                func(resources, wrapper.item);
            }
        }
    }
}

struct AppLoop {
    world: World,
    resources: Resources,
    schedule: Wrapper<Schedule>,
    events: Wrapper<Receiver<LoopEvent>>,
    mtx: Arc<Mutex<bool>>,
    barrier: Arc<Barrier>,
    on_schedule_start: Vec<fn(&mut World, &mut Resources, &Receiver<LoopEvent>)>,
    on_schedule_end: Vec<fn(&mut World, &mut Resources, &Receiver<LoopEvent>)>,
}

impl AppLoop {
    fn start(mut app: Arc<AppLoop>, pool: &ThreadPool) {
        pool.spawn(move || {
            let app = unsafe { Arc::get_mut_unchecked(&mut app) };

            loop {
                let _guard = app.mtx.lock();
                
                for func in app.on_schedule_start.iter() {
                    func(&mut app.world, &mut app.resources, &app.events.item);
                }

                app.schedule.item.execute(&mut app.world, &mut app.resources);

                for func in app.on_schedule_end.iter() {
                    func(&mut app.world, &mut app.resources, &app.events.item);
                }
            }

            app.barrier.wait();
        });
    }
}

struct SysLoop {
    world: World,
    resources: Resources,
    schedule: Wrapper<Schedule>,
    events: Wrapper<Receiver<LoopEvent>>,
    mtx: Arc<Mutex<bool>>,
    barrier: Arc<Barrier>,
    on_schedule_start: Vec<fn(&mut World, &mut Resources, &Receiver<LoopEvent>)>,
    on_schedule_end: Vec<fn(&mut World, &mut Resources, &Receiver<LoopEvent>)>,
    on_schedule_wait: Vec<fn(&mut World, &mut Resources, &Receiver<LoopEvent>)>,
    update: fn(&mut AppLoop, &SysLoop),
    run: Arc<AtomicBool>,
}

impl SysLoop {
    fn start(mut sys: Arc<SysLoop>, mut app: Arc<AppLoop>, pool: &ThreadPool) {
        pool.spawn(move || {
            let sys = unsafe { Arc::get_mut_unchecked(&mut sys) };

            loop {
                if sys.run.load(Ordering::Relaxed) {
                    sys.run.store(false, Ordering::Relaxed);

                    for func in sys.on_schedule_start.iter() {
                        func(&mut sys.world, &mut sys.resources, &sys.events.item);
                    }
        
                    sys.schedule.item.execute(&mut sys.world, &mut sys.resources);
        
                    for func in sys.on_schedule_end.iter() {
                        func(&mut sys.world, &mut sys.resources, &sys.events.item);
                    }
        
                    let _guard = sys.mtx.lock();
        
                    let app = unsafe { Arc::get_mut_unchecked(&mut app) };
        
                    (sys.update)(app, &sys);
                } else {
                    for func in sys.on_schedule_wait.iter() {
                        func(&mut sys.world, &mut sys.resources, &sys.events.item);
                    }
                }
            }

            sys.barrier.wait();
        });
    }
}

struct Core {
    universe: Universe,
    app: Arc<AppLoop>,
    sys: Arc<SysLoop>,
    pools: Vec<ThreadPool>,
    barrier: Arc<Barrier>,
}

impl Core {
    fn new() -> Self {
        let universe = Universe::new();
        let barrier = Arc::new(Barrier::new(3));
        let mtx = Arc::new(Mutex::new(false));
        let run = Arc::new(AtomicBool::new(false));
        let pools = vec![ThreadPoolBuilder::new().num_threads(1).build().unwrap(), ThreadPoolBuilder::new().num_threads(num_cpus::get() - 1).build().unwrap()];

        let (producer_app, consumer_app) = channel::<LoopEvent>();
        let (producer_sys, consumer_sys) = channel::<LoopEvent>();

        let mut resources_app = Resources::default();
        let mut resources_sys = Resources::default();

        resources_app.insert(Wrapper { item: producer_sys });
        resources_sys.insert(Wrapper { item: producer_app });

        resources_app.insert(run.clone());

        let app = AppLoop {
            world: universe.create_world(),
            resources: resources_app,
            schedule: Wrapper { item: Schedule::builder().build() },
            events: Wrapper { item: consumer_app },
            mtx: mtx.clone(),
            barrier: barrier.clone(),
            on_schedule_start: Vec::new(),
            on_schedule_end: Vec::new(),
        };
        let sys = SysLoop {
            world: universe.create_world(),
            resources: resources_sys,
            schedule: Wrapper { item: Schedule::builder().build() },
            events: Wrapper { item: consumer_sys },
            mtx: mtx.clone(),
            barrier: barrier.clone(),
            on_schedule_start: Vec::new(),
            on_schedule_end: Vec::new(),
            on_schedule_wait: Vec::new(),
            update: |_, _| {},
            run: run.clone(),
        };

        let app = Arc::new(app);
        let sys = Arc::new(sys);

        Core {
            universe,
            app,
            sys,
            pools,
            barrier,
        }
    }

    fn load_pixels(&mut self) {
        let mut map = map::ProvBuilder::new(1024, 0.1, 0.6, 2., 0., 1., 0.1, 0.9, -60., 20.);

        map.gen_heightmap();
        map.gen_insolation();
        map.gen_waters();
        map.gen_cloud();
        map.gen_temp();
        map.gen_rivermap();
        map.gen_watermap();
        map.gen_vegetmap();

        map.export(&map.heightmap, "heightmap.png");
        map.export_minmax(&map.insolation, "insolation.png", 0., 1.);
        map.export_waters("waters.png");
        map.export_minmax(&map.cloudmap, "cloudmap.png", 0., 1.);
        map.export_minmax(&map.tempmap, "tempmap.png", 0., 1.);
        map.export(&map.rivermap, "rivermap.png");
        map.export(&map.watermap, "watermap.png");
        map.export(&map.vegetmap, "vegetmap.png");

        let world = unsafe { &mut Arc::get_mut_unchecked(&mut self.sys).world };

        let pixels = world.insert(
            (Pixel,),
            (0..map.size * map.size).map(|i| {
                (
                    Height { item: map.heightmap[i] },
                    Heat { item: map.tempmap[i] },
                    Water { item: map.watermap[i] },
                    River { item: map.rivermap[i] },
                    Rain { item: map.cloudmap[i] },
                    Veget { item: map.vegetmap[i] },
                )
            })
        ).to_vec();

        for (i, &pixel) in pixels.iter().enumerate() {
            let neighb = map.neighbs[i].iter().map(|&(i, _)| pixels[i]).collect();

            world.add_component(pixel, Neighb { item: neighb }).unwrap();

            if let Some(water) = map.waters.get(&i) {
                match water {
                    map::Water::Sea => world.add_tag(pixel, Sea).unwrap(),
                    map::Water::Lake => world.add_tag(pixel, Lake).unwrap(),
                };
            }
        }
    }

    fn start(&mut self) {
        AppLoop::start(self.app.clone(), &self.pools[0]);
        SysLoop::start(self.sys.clone(), self.app.clone(), &self.pools[1]);

        self.barrier.wait();
    }
}

fn main() {
    let mut core = Core::new();

    core.load_pixels();
}

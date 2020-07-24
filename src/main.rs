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

use half::f16;

use serde::Deserialize;
use ron::de::from_reader;

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

enum BuildingAction {
}

#[derive(Clone, Deserialize)]
struct Defines {
    size: usize,
    building_i: HashMap<String, usize>,
    land_i: HashMap<String, usize>,
}

#[derive(Clone, Deserialize)]
struct BuildingDefines {
    name: String,
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct Pixel;
#[derive(Clone, Copy, Debug, PartialEq)]
struct Lake;
#[derive(Clone, Copy, Debug, PartialEq)]
struct Sea;
#[derive(Clone, Copy, Debug, PartialEq)]
struct Settlement;
#[derive(Clone, Copy, Debug, PartialEq)]
struct Colony;

struct Owned { item: Entity }
struct Owns { item: Vec<Entity> }
struct Location { item: Entity }
struct Name { item: String }
struct Water { item: f32 }
struct River { item: f32 }
struct Rain { item: f32 }
struct Heat { item: f32 }
struct Height { item: f32 }
struct Veget { item: f32 }
struct Neighb { item: Vec<Entity> }
struct RiverBase { item: f32 }
struct VegetBase { item: f32 }
struct Speed { item: f32 }
struct Capacity { item: f32 }
struct Index { item: usize }
struct Pop { item: f32 }
struct Skill { item: Vec<f32> }
struct Building { item: Vec<f32> }
struct Land { item: Vec<f32> }

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
    defines: Defines,
}

impl Core {
    fn new() -> Self {
        let universe = Universe::new();
        let barrier = Arc::new(Barrier::new(3));
        let mtx = Arc::new(Mutex::new(false));
        let run = Arc::new(AtomicBool::new(false));
        let pools = vec![ThreadPoolBuilder::new().num_threads(1).build().unwrap(), ThreadPoolBuilder::new().num_threads(num_cpus::get() - 1).build().unwrap()];
        let defines: Defines = from_reader(File::open("defines.ron").unwrap()).unwrap();

        let (producer_app, consumer_app) = channel::<LoopEvent>();
        let (producer_sys, consumer_sys) = channel::<LoopEvent>();

        let mut resources_app = Resources::default();
        let mut resources_sys = Resources::default();

        resources_app.insert(Wrapper { item: producer_sys });
        resources_sys.insert(Wrapper { item: producer_app });

        resources_app.insert(run.clone());

        resources_app.insert(defines.clone());
        resources_sys.insert(defines.clone());

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
            defines,
        }
    }

    fn load_pixels(&mut self) {
        let mut map = map::ProvBuilder::new(self.defines.size, 0.1, 0.6, 2., 0., 1., 0.1, 0.9, -20., -10.);

        map.gen_heightmap();
        map.gen_insolation();
        map.gen_waters();
        map.gen_cloud();
        map.gen_temp();
        map.gen_rivermap();
        map.gen_watermap();
        map.gen_vegetmap();
        map.gen_settlements();

        map.export(&map.heightmap, "heightmap.png");
        map.export_minmax(&map.insolation, "insolation.png", 0., 1.);
        map.export_waters("waters.png");
        map.export_minmax(&map.cloudmap, "cloudmap.png", 0., 1.);
        map.export_minmax(&map.tempmap, "tempmap.png", 0., 1.);
        map.export_minmax(&map.rivermap, "rivermap.png", 0., 1.);
        map.export_minmax(&map.watermap, "watermap.png", 0., 1.);
        map.export_minmax(&map.vegetmap, "vegetmap.png", 0., 1.);
        map.export_settlements("settlements.png");

        let world = unsafe { &mut Arc::get_mut_unchecked(&mut self.sys).world };

        let pixels = world.insert(
            (Pixel,),
            (0..map.size * map.size).map(|i| {
                (
                    Height { item: map.heightmap[i] as f32 },
                    Heat { item: map.tempmap[i] as f32 },
                    River { item: map.rivermap[i] as f32 },
                    Rain { item: map.cloudmap[i] as f32 },
                    Veget { item: map.vegetmap[i] as f32 },
                    Water { item: map.watermap[i] as f32 },
                    RiverBase { item: map.rivermap[i] as f32 },
                    VegetBase { item: map.vegetmap[i] as f32 },
                    Building { item: Vec::new() },
                    Land { item: Vec::new() },
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
            if map.settlements[i] {
                world.add_tag(pixel, Settlement).unwrap();
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

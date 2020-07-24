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

static SET: fn(&mut f32, &f32) = |x, y| *x = *y;
static MAX: fn(&mut f32, &f32) = |x, y| *x = max_by(*x, *y, |x, y| x.partial_cmp(y).unwrap());
static MIN: fn(&mut f32, &f32) = |x, y| *x = min_by(*x, *y, |x, y| x.partial_cmp(y).unwrap());
static ADD: fn(&mut f32, &f32) = |x, y| *x += y;
static SUBT: fn(&mut f32, &f32) = |x, y| *x -= y;
static MULT: fn(&mut f32, &f32) = |x, y| *x *= y;
static DIV: fn(&mut f32, &f32) = |x, y| *x /= y;
static POW: fn(&mut f32, &f32) = |x, y| *x = x.powf(*y);
static ROOT: fn(&mut f32, &f32) = |x, y| *x = x.powf(1./y);
static LOG: fn(&mut f32, &f32) = |x, y| *x = x.log(*y);
static FUNCS: [fn(&mut f32, &f32); 10] = [
    SET,
    MAX,
    MIN,
    ADD,
    SUBT,
    MULT,
    DIV,
    POW,
    ROOT,
    LOG,
];

pub fn get_func(name: &String) -> u8 {
    match name.as_str() {
        "SET" => 0,
        "MAX" => 1,
        "MIN" => 2,
        "ADD" => 3,
        "SUBT" => 4,
        "MULT" => 5,
        "DIV" => 6,
        "POW" => 7,
        "ROOT" => 8,
        "LOG" => 9,
        _ => 10,
    }
}

struct Value<'s> {
    name: &'s str,
    handle: Arc<ValueHandle>,
    base: f32,
    paras: Option<u16>,
}

struct ValueHandle {
    value: f32,
    change: f32,
}

struct ValueManager<'s> {
    values: Vec<Value<'s>>,
    paras: Vec<Vec<(u8, u8)>>
}

impl<'s> ValueManager<'s> {
    fn new() -> Self {
        let mut value = ValueManager {
            values: Vec::new(),
            paras: Vec::new(),
        };

        value.add_value("0", 0., Vec::<&str>::new(), Vec::<&str>::new());
        value.add_value("1", 1., Vec::<&str>::new(), Vec::<&str>::new());
        value.add_value("-1", -1., Vec::<&str>::new(), Vec::<&str>::new());
        value.add_value("2", 2., Vec::<&str>::new(), Vec::<&str>::new());
        value.add_value("-2", -2., Vec::<&str>::new(), Vec::<&str>::new());

        return value;
    }

    fn add_value<T: Into<String> + Copy>(&mut self, name: &'s str, base: f32, funcs: Vec<T>, parents: Vec<T>) -> Arc<ValueHandle> {
        let funcs: Vec<String> = funcs.iter().map(|&f| f.into()).collect();
        let parents: Vec<String> = parents.iter().map(|&p| p.into()).collect();

        let paras = match funcs.len() {
            0 => None,
            _ => {
                let mut paras = Vec::new();

                for i in 0..funcs.len() {
                    for (ii, value) in self.values.iter().enumerate() {
                        if value.name == parents[i] {
                            paras.push((get_func(&funcs[i]), ii as u8));

                            break;
                        }
                    }
                }

                self.paras.push(paras);
                Some(self.paras.len() as u16 - 1)
            }
        };

        self.values.push(Value {
            name,
            handle: Arc::new(ValueHandle { value: base, change: 0. }),
            base,
            paras,
        });

        return self.values.last().unwrap().handle.clone();
    }

    fn update(&mut self) {
        let mut update = Vec::new();

        for (i, value) in self.values.iter_mut().enumerate() {
            let handle = unsafe { Arc::get_mut_unchecked(&mut value.handle) };

            if handle.change != 0. {
                update.push(i);

                value.base += handle.change;
                handle.change = 0.;
            }
        }

        let mut stack = update.clone();

        while !stack.is_empty() {
            let i = stack.pop().unwrap();

            for (ii, value) in self.values.iter().enumerate() {
                if let Some(paras) = &value.paras {
                    for (_, parent) in self.paras[*paras as usize].iter() {
                        if i as u8 == *parent && !update.contains(&ii) {
                            update.push(ii);
                            stack.push(ii);

                            break;
                        }
                    }
                }
            }
        }

        while !update.is_empty() {
            let mut is = Vec::new();

            'outer: for ii in (0..update.len()).rev() {
                let i = update[ii].clone();

                if let Some(paras) = &self.values[i].paras {
                    for (_, parent) in self.paras[*paras as usize].iter() {
                        if update.contains(&(*parent as usize)) {
                            continue 'outer;
                        }
                    }
                }

                is.push(i);
                update.remove(ii);
            }

            for &i in is.iter() {
                let mut value = self.values[i].base;

                if let Some(paras) = &self.values[i].paras {
                    for (func, parent) in self.paras[*paras as usize].iter() {
                        FUNCS[*func as usize](&mut value, &self.values[*parent as usize].handle.value);
                    }
                }

                unsafe { Arc::get_mut_unchecked(&mut self.values[i].handle).value = value; }
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
#[derive(Clone, Copy, Debug, PartialEq)]
struct Settlement;
#[derive(Clone, Copy, Debug, PartialEq)]
struct Colony;

struct Owned { item: Entity }
struct Owns { item: Vec<Entity> }
struct Name { item: String }
struct Water { value: Arc<ValueHandle> }
struct River { value: Arc<ValueHandle> }
struct Rain { value: Arc<ValueHandle> }
struct Heat { value: Arc<ValueHandle> }
struct Height { value: Arc<ValueHandle> }
struct Veget { value: Arc<ValueHandle> }
struct Neighb { item: Vec<Entity> }
struct RiverBase { item: f32 }
struct VegetBase { item: f32 }
struct Pop { value: Arc<ValueHandle> }
struct Skill { item: HashMap<String, Arc<ValueHandle>> }
struct Building { item: HashMap<String, Arc<ValueHandle>> }
struct Land { item: HashMap<String, Arc<ValueHandle>> }

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
        map.export_minmax(&map.rivermap, "rivermap.png", 0., 1.);
        map.export_minmax(&map.watermap, "watermap.png", 0., 1.);
        map.export_minmax(&map.vegetmap, "vegetmap.png", 0., 1.);

        let world = unsafe { &mut Arc::get_mut_unchecked(&mut self.sys).world };

        let pixels = world.insert(
            (Pixel,),
            (0..map.size * map.size).map(|i| {
                let mut manager = ValueManager::new();

                let height = manager.add_value("Height", map.heightmap[i] as f32, Vec::<&str>::new(), Vec::<&str>::new());
                let heat = manager.add_value("Heat", map.tempmap[i] as f32, Vec::<&str>::new(), Vec::<&str>::new());
                let river = manager.add_value("River", map.rivermap[i] as f32, Vec::<&str>::new(), Vec::<&str>::new());
                let rain = manager.add_value("Rain", map.cloudmap[i] as f32, Vec::<&str>::new(), Vec::<&str>::new());
                let veget = manager.add_value("Veget", map.vegetmap[i] as f32, Vec::<&str>::new(), Vec::<&str>::new());
                let water = manager.add_value("Water", 0., vec!["SET", "ADD", "DIV"], vec!["River", "Rain", "2"]);

                (
                    manager,
                    Height { value: height },
                    Heat { value: heat },
                    River { value: river },
                    Rain { value: rain },
                    Veget { value: veget },
                    Water { value: water },
                    RiverBase { item: map.rivermap[i] as f32 },
                    VegetBase { item: map.vegetmap[i] as f32 },
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

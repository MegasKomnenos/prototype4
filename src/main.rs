use legion::prelude::*;
use legion::systems::resource::Resources;
use legion::systems::schedule::Schedule;
use rayon::prelude::*;
use rayon::ThreadPool;
use rayon::ThreadPoolBuilder;

use std::collections::HashMap;
use std::sync::Barrier;

struct ScheduleWrapper {
    schedule: Schedule,
}

unsafe impl Send for ScheduleWrapper {}

struct Loop {
    name: String,
    running: bool,
    waiting: bool,
    world: World,
    resources: Resources,
    schedule: ScheduleWrapper,
}

impl Loop {
    fn new(name: String, world: World, resources: Resources, schedule: Schedule) -> Self {
        Loop {
            name,
            running: false,
            waiting: false,
            world,
            resources,
            schedule: ScheduleWrapper { schedule },
        }
    }

    fn execute(&mut self, pool: &mut ThreadPool) {
        if !self.running && !self.waiting {
            self.running = true;

            pool.install(|| {
                self.schedule.schedule.execute(&mut self.world, &mut self.resources);
                self.running = false;
            });
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

struct Application {
    loops: Vec<Loop>,
    pools: Vec<ThreadPool>,
    links: Vec<Link>,
    name_id: HashMap<String, usize>,
}

impl Application {
    fn new() -> Self {
        Application {
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
                if !link.waiting {
                    let from_id = link.from_id.unwrap();

                    if !self.loops[from_id].running {
                        link.waiting = true;
                        self.loops[from_id].waiting = true;
                    }
                } else {
                    let to_id = link.to_id.unwrap();

                    if !self.loops[to_id].running {
                        let from_id = link.from_id.unwrap();

                        link.waiting = false;
                        self.loops[from_id].waiting = false;

                        (link.update)(&mut self.loops, from_id, to_id);
                    }
                }
            }

            for (i, lp) in self.loops.iter_mut().enumerate() {
                lp.execute(&mut self.pools[i]);
            }
        }
    }
}

fn main() {
}

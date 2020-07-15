use legion::prelude::*;
use legion::entity::Entity;
use legion::systems::SystemBuilder;
use legion::systems::resource::Resources;
use legion::systems::schedule::Schedule;
use rayon::prelude::*;
use rayon::ThreadPool;
use rayon::ThreadPoolBuilder;

use std::collections::HashMap;

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

    let mut app = Application::new();

    app.add_loop(loop0, 1);
    app.add_loop(loop1, 1);
    app.add_link(link);

    app.execute();
}

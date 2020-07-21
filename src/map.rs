use noise::Perlin;
use noise::NoiseFn;
use image::RgbImage;
use image::Rgb;

use pathfinding::directed::dijkstra::dijkstra;

use rand::Rng;

use std::mem::swap;
use std::cmp::Ordering;
use std::collections::HashMap;
use std::collections::BinaryHeap;
use std::path::PathBuf;

struct PerlinOctave {
    noise: Perlin,
    size: usize,
    octave: usize,
    freq: f64,
    pers: f64,
    lac: f64,
    min: f64,
    max: f64,
}

impl PerlinOctave {
    fn new(size: usize, freq: f64, pers: f64, lac: f64, min: f64, max: f64) -> Self {
        PerlinOctave {
            noise: Perlin::new(),
            size,
            octave: (size as f64).log2() as usize,
            freq,
            pers,
            lac,
            min,
            max,
        }
    }

    fn get(&self, x: f64, y: f64) -> f64 {
        let max = (2f64).sqrt() / 2.;

        let mut out = 0.;
        let mut max_total = 0.;
        let mut amp = 1.;

        let mut x = x * self.freq / (self.size as f64).sqrt();
        let mut y = y * self.freq / (self.size as f64).sqrt();

        for _ in 0..self.octave {
            out += self.noise.get([x, y]) * amp;
            max_total += max * amp;

            amp *= self.pers;
            x *= self.lac;
            y *= self.lac;
        }

        out += max_total;
        out /= max_total;
        out /= 2.;
        out *= self.max - self.min;
        out += self.min;

        out
    }
}

pub enum Water {
    Sea,
    Lake
}

pub struct ProvBuilder {
    noise: PerlinOctave,
    pub heightmap: Vec<f64>,
    pub tempmap: Vec<f64>,
    pub rainmap: Vec<f64>,
    pub rivermap: Vec<f64>,
    pub waters: HashMap<usize, Water>,
    water_level: f64,
    water_taper: f64,
    temp_base: f64,
    temp_height: f64,
    temp_latitude: f64,
    rain_wind: (f64, f64),
    rain_height: f64,
    rain_fall: f64,
}

impl ProvBuilder {
    pub fn new(
        size: usize, freq: f64, pers: f64, lac: f64, min: f64, max: f64, water_level: f64, water_taper: f64, 
        temp_base: f64, temp_height: f64, temp_latitude: f64,
        rain_wind: (f64, f64), rain_height: f64, rain_fall: f64,
    ) -> Self {
        let noise = PerlinOctave {
            noise: Perlin::new(),
            size,
            octave: (size as f64).log2() as usize,
            freq,
            pers,
            lac,
            min,
            max,
        };

        ProvBuilder {
            noise,
            heightmap: Vec::new(),
            tempmap: Vec::new(),
            rainmap: Vec::new(),
            rivermap: Vec::new(),
            waters: HashMap::new(),
            water_level,
            water_taper,
            temp_base,
            temp_height,
            temp_latitude,
            rain_wind,
            rain_height,
            rain_fall,
        }
    }

    pub fn gen_heightmap(&mut self) {
        let size = self.noise.size;
        let med = size / 2;

        self.heightmap.clear();
        self.heightmap.reserve_exact(size * size);

        for y in 0..size {
            for x in 0..size {
                let dist = (((med as f64 - x as f64).powi(2) + (med as f64 - y as f64).powi(2)).sqrt() / med as f64).powi(2);
                let val = match self.noise.get(x as f64, y as f64) - self.water_level - self.water_taper * dist {
                    x if x >= 0. => x / (1. - self.water_level),
                    _ => 0.,
                };

                self.heightmap.push(val * val);
            }
        }
    }

    pub fn gen_waters(&mut self) {
        let size = self.noise.size;
        let mut stack = Vec::new();

        stack.push(0);
        self.waters.clear();

        while !stack.is_empty() {
            let i = stack.pop().unwrap();

            if self.waters.contains_key(&i) || self.heightmap[i] > 0. {
                continue;
            }

            self.waters.insert(i, Water::Sea);

            if (i + 1) % size > 0 {
                stack.push(i + 1);
            }
            if i % size > 0 {
                stack.push(i - 1);
            }
            if i >= size {
                stack.push(i - size);
            }
            if i + size < size * size {
                stack.push(i + size);
            }
        }

        for (i, &height) in self.heightmap.iter().enumerate() {
            if height == 0. && !self.waters.contains_key(&i) {
                self.waters.insert(i, Water::Lake);
            }
        }
    }

    pub fn gen_tempmap(&mut self) {
        let size = self.noise.size;

        self.tempmap.clear();
        self.tempmap.reserve_exact(size * size);

        let mut i = 0;

        for y in 0..size {
            for _ in 0..size {
                let val = self.temp_base + (y as f64 / size as f64) * self.temp_latitude - self.heightmap[i] * self.temp_height;

                self.tempmap.push(val);

                i += 1;
            }
        }
    }

    pub fn gen_rainmap(&mut self) {
        let size = self.noise.size;
        let sizei = size as isize;

        let mut flows = Vec::new();

        if self.rain_wind.0 > 0. {
            if self.rain_wind.1 >= 0. {
                let upper = self.rain_wind.1 / (self.rain_wind.0 + self.rain_wind.1);
                let upper_right = 1. - (self.rain_wind.0 - self.rain_wind.1).abs() / (self.rain_wind.0 + self.rain_wind.1);
                let right = self.rain_wind.0 / (self.rain_wind.0 + self.rain_wind.1);
                let sm = upper + upper_right + right;

                flows.push((sizei, upper / sm));
                flows.push((sizei + 1, upper_right / sm));
                flows.push((1, right / sm));
            } else if self.rain_wind.1 == 0. {
                flows.push((1, 1.));
            } else {
                let lower = -self.rain_wind.1 / (self.rain_wind.0 - self.rain_wind.1);
                let lower_right = 1. - (self.rain_wind.0 + self.rain_wind.1).abs() / (self.rain_wind.0 - self.rain_wind.1);
                let right = self.rain_wind.0 / (self.rain_wind.0 - self.rain_wind.1);
                let sm = lower + lower_right + right;

                flows.push((-sizei, lower / sm));
                flows.push((-sizei + 1, lower_right / sm));
                flows.push((1, right / sm));
            }
        } else if self.rain_wind.0 < 0. {
            if self.rain_wind.1 >= 0. {
                let upper = self.rain_wind.1 / (-self.rain_wind.0 + self.rain_wind.1);
                let upper_left = 1. - (-self.rain_wind.0 - self.rain_wind.1).abs() / (-self.rain_wind.0 + self.rain_wind.1);
                let left = -self.rain_wind.0 / (-self.rain_wind.0 + self.rain_wind.1);
                let sm = upper + upper_left + left;

                flows.push((sizei, upper / sm));
                flows.push((sizei - 1, upper_left / sm));
                flows.push((-1, left / sm));
            } else if self.rain_wind.1 == 0. {
                flows.push((-1, 1.));
            } else {
                let lower = -self.rain_wind.1 / (-self.rain_wind.0 - self.rain_wind.1);
                let lower_left = 1. - (-self.rain_wind.0 + self.rain_wind.1).abs() / (-self.rain_wind.0 - self.rain_wind.1);
                let left = -self.rain_wind.0 / (-self.rain_wind.0 - self.rain_wind.1);
                let sm = lower + lower_left + left;

                flows.push((-sizei, lower / sm));
                flows.push((-sizei - 1, lower_left / sm));
                flows.push((-1, left / sm));
            }
        } else {
            if self.rain_wind.1 > 0. {
                flows.push((sizei, 1.));
            } else if self.rain_wind.1 < 0. {
                flows.push((-sizei, 1.));
            }
        }

        let mut moist = vec![0.; size * size];

        let mut i = 0;

        for y in 0..size {
            for x in 0..size {
                if x <= 1 || x + 2 >= size || y <= 1 || y + 2 >= size {
                    moist[i] = 1.;
                }

                i += 1;
            }
        }

        for _ in 0..size*2 {
            self.rainmap = vec![0.; size * size];
            let mut moist_new = vec![0.; size * size];

            let mut i = 0;

            for y in 0..size {
                for x in 0..size {
                    if x == 0 || x + 1 == size || y == 0 || y + 1 == size {
                        i += 1;

                        continue;
                    }

                    let rain = moist[i] * self.rain_fall;

                    self.rainmap[i] += rain;
                    moist[i] -= rain;

                    for flow in flows.iter() {
                        let ii = (i as isize + flow.0) as usize;
                        let moist_move = moist[i] * flow.1;
                        let rain = match 2f64.powf(self.heightmap[ii] - self.heightmap[i]) - 1. {
                            x if x > 0. => x * self.rain_height,
                            _ => 0.,
                        };

                        moist_new[ii] += moist_move * (1. - rain);
                        self.rainmap[ii] += moist_move * rain;
                    }

                    i += 1;
                }
            }

            let mut i = 0;

            for y in 0..size {
                for x in 0..size {
                    if x <= 1 || x + 2 >= size || y <= 1 || y + 2 >= size {
                        moist_new[i] = 1.;
                    }

                    i += 1;
                }
            }

            swap(&mut moist, &mut moist_new);
        }

        let max = self.rainmap.iter().max_by(|x, y| x.partial_cmp(y).unwrap()).unwrap();
        let min = self.rainmap.iter().min_by(|x, y| x.partial_cmp(y).unwrap()).unwrap();
        self.rainmap = self.rainmap.iter().map(|x| (x - min) / (max - min)).collect();
    }

    pub fn gen_rivermap(&mut self) {
        let size = self.noise.size;

        let choices = [
            (1, 1.),
            (-1 as isize, 1.),
            (size as isize, 1.),
            (-(size as isize), 1.),
            (1 + size as isize, 2f64.sqrt()),
            (-1 + size as isize, 2f64.sqrt()),
            (1 - (size as isize), 2f64.sqrt()),
            (-1 - (size as isize), 2f64.sqrt()),
        ];

        let mut river_drainage = vec![0; size * size];
        let mut height_ordered = self.heightmap
            .iter()
            .enumerate()
            .filter(|(_, &height)| height > 0.)
            .collect::<Vec<(usize, &f64)>>();
        height_ordered.sort_by(|(_, a), (_, b)| b.partial_cmp(a).unwrap());

        loop {
            if let Some((i, _)) = height_ordered.pop() {
                let (paths, _) = dijkstra(&i, 
                    |&i| {
                        let choices: Vec<(usize, usize)> = choices
                            .iter()
                            .map(|(ii, c)| {
                                let ii = (i as isize + ii) as usize;
                                (ii, (10000. * c * (self.heightmap[ii] / (self.heightmap[i] + 0.001))) as usize)
                            })
                            .collect();
                        
                        return choices;
                    },
                    |&i| {
                        if river_drainage[i] != 0 {
                            return true;
                        } else if let Some(water) = self.waters.get(&i) {
                            match water {
                                Water::Lake => return false,
                                Water::Sea => return true,
                            };
                        }

                        return false;
                    }
                ).unwrap();
                
                for (i, &path) in paths.iter().enumerate() {
                    if i + 1 < paths.len() {
                        river_drainage[path] = paths[i + 1];
                    }
                }
            } else {
                break;
            }
        }

        self.rivermap = vec![0.; size * size];

        for _ in 0..size {
            let mut rivermap_new = vec![0.; size * size];

            for (i, &rain) in self.rainmap.iter().enumerate() {
                if river_drainage[i] > 0 {
                    rivermap_new[i] += rain;
                }
            }
            for (i, &river) in river_drainage.iter().enumerate() {
                if river > 0 {
                    rivermap_new[river] += self.rivermap[i];
                }
            }

            swap(&mut self.rivermap, &mut rivermap_new);
        }

        for river in self.rivermap.iter_mut() {
            *river = river.sqrt();
        }

        let mx = self.rivermap
            .iter()
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap()
            .clone();
        
        for river in self.rivermap.iter_mut() {
            *river /= mx;
        }
    }

    pub fn export<T: Into<PathBuf>>(&self, map: &Vec<f64>, path: T) {
        let mut i = 0;
        let mut img = RgbImage::new(self.noise.size as u32, self.noise.size as u32);

        let max = map.iter().max_by(|x, y| x.partial_cmp(y).unwrap()).unwrap();
        let min = map.iter().min_by(|x, y| x.partial_cmp(y).unwrap()).unwrap();
        let map: Vec<f64> = map.iter().map(|x| (x - min) / (max - min)).collect();

        for y in 0..self.noise.size {
            for x in 0..self.noise.size {
                let val = (map[i] * 255.) as u8;
                
                i += 1;

                img.put_pixel(x as u32, y as u32, Rgb([val, val, val]));
            }
        }

        img.save(path.into()).unwrap();
    }

    pub fn export_waters<T: Into<PathBuf>>(&self, path: T) {
        let mut i = 0;
        let mut img = RgbImage::new(self.noise.size as u32, self.noise.size as u32);

        for y in 0..self.noise.size {
            for x in 0..self.noise.size {
                if let Some(water) = self.waters.get(&i) {
                    match water {
                        Water::Lake => img.put_pixel(x as u32, y as u32, Rgb([128, 128, 128])),
                        Water::Sea => img.put_pixel(x as u32, y as u32, Rgb([255, 255, 255])),
                    }
                } else {
                    img.put_pixel(x as u32, y as u32, Rgb([0, 0, 0]));
                }

                i += 1;
            }
        }

        img.save(path.into()).unwrap();
    }
}
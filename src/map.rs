use noise::Perlin;
use noise::NoiseFn;
use image::RgbImage;
use image::Rgb;
use num::clamp;

use pathfinding::directed::dijkstra::dijkstra;

use rand::Rng;

use std::mem::swap;
use std::collections::HashMap;
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

pub fn is_neighbor(i: usize, ii: usize, size: usize) -> bool {
    let x_i = i % size;
    let y_i = i / size;

    let x_ii = ii % size;
    let y_ii = ii / size;

    let x_diff = match x_i >= x_ii {
        true => x_i - x_ii,
        false => x_ii - x_i,
    };
    let y_diff = match y_i >= y_ii {
        true => y_i - y_ii,
        false => y_ii - y_i,
    };

    return x_diff <= 1 && y_diff <= 1 && y_ii < size;
}

fn find_lat(lats: &Vec<f64>, targ: f64, size: usize) -> usize {
    let mut prev = 0;
    let mut prev_diff = f64::MAX;

    for y in 0..size {
        let lat = lats[y * size];
        let diff = (lat - targ).abs();

        if diff < prev_diff {
            prev = y;
            prev_diff = diff;
        } else {
            return prev;
        }
    }

    return prev;
} 

fn do_wind(x: usize, y: usize, y_to: usize, lat: f64, lat_goal: f64, 
    flow: (f64, f64), size: usize, 
    cloudmap: &mut Vec<f64>, latitudes: &Vec<f64>, heightmap: &Vec<f64>, 
    water_gain: f64, water_mult: f64) 
{
    let mut flow_t = flow;

    if flow_t.0.abs() != 1. {
        flow_t.0 /= flow_t.0.abs();
        flow_t.1 /= flow_t.0.abs();
    }

    let mut line = Vec::new();
    let mut res = 0.;
    let mut x_t = x;
    let mut y_t = y;

    while y_t != y_to {
        if x_t < size {
            line.push(x_t + y_t * size);
        }

        x_t = (x_t as isize + flow_t.0 as isize) as usize;
        res += flow_t.1.abs();

        while res >= 1. {
            res -= 1.;
            y_t = (y_t as isize + flow_t.1.signum() as isize) as usize;

            if x_t < size && res >= 1. {
                line.push(x_t + y_t * size);
            }
            if y_t == y_to {
                break;
            }
        }
    }

    let mut water = size as f64 * water_mult / 100.;

    for &ii in line.iter() {
        water += water_gain;

        let lat_t = latitudes[ii];

        let cloud = water * (heightmap[ii].powf(0.75) + ((lat_t - lat) / (lat_goal - lat)).powi(3)) * 50. / size as f64;
        cloudmap[ii] = cloud;
        water -= cloud;
    }
}

pub struct ProvBuilder {
    noise: PerlinOctave,
    pub neighbs: Vec<Vec<(usize, f64)>>,
    pub heightmap: Vec<f64>,
    pub waters: HashMap<usize, Water>,
    pub insolation: Vec<f64>,
    pub latitude: Vec<f64>,
    pub cloudmap: Vec<f64>,
    pub rivermap: Vec<f64>,
    pub tempmap: Vec<f64>,
    pub watermap: Vec<f64>,
    pub vegetmap: Vec<f64>,
    water_level: f64,
    water_taper: f64,
    lat_start: f64,
    lat_end: f64,
}

impl ProvBuilder {
    pub fn new(
        size: usize, freq: f64, pers: f64, lac: f64, min: f64, max: f64, water_level: f64, water_taper: f64, 
        lat_start: f64, lat_end: f64,
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

        let neighbs = (0..size * size)
            .into_iter()
            .map(|i| {
                choices
                    .iter()
                    .map(|&(ii, c)| ((i as isize + ii) as usize, c))
                    .filter(|&(ii, _)| is_neighbor(i, ii, size))
                    .collect()
            })
            .collect();

        ProvBuilder {
            noise,
            neighbs,
            heightmap: Vec::new(),
            waters: HashMap::new(),
            insolation: Vec::new(),
            latitude: Vec::new(),
            cloudmap: Vec::new(),
            rivermap: Vec::new(),
            tempmap: Vec::new(),
            watermap: Vec::new(),
            vegetmap: Vec::new(),
            water_level,
            water_taper,
            lat_start,
            lat_end,
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

    pub fn gen_insolation(&mut self) {
        let size = self.noise.size;

        self.latitude = vec![0.; size * size];
        self.insolation = vec![0.; size * size];

        for y in 0..size {
            let lat = (self.lat_end - self.lat_start) * y as f64 / (size - 1) as f64 + self.lat_start;
            let insolation = -lat.powi(2) / 10000. + 1.;

            for x in 0..size {
                self.latitude[x + y * size] = lat;
                self.insolation[x + y * size] = insolation;
            }
        }
    }

    pub fn gen_cloud(&mut self) {
        let size = self.noise.size;

        self.cloudmap = vec![0.; size * size];

        let s60 = find_lat(&self.latitude, -60., size);
        let s30 = find_lat(&self.latitude, -30., size);
        let s0 = find_lat(&self.latitude, 0., size);
        let n30 = find_lat(&self.latitude, 30., size);
        let n60 = find_lat(&self.latitude, 60., size);

        if s30 != s60 {
            for x in 0..size {
                do_wind(x, s30, s60, -30., -60., (1., -1.), size, &mut self.cloudmap, &self.latitude, &self.heightmap, 0.2, 1.);
            }
            for y in s60..s30 {
                do_wind(0, y, s60, -30., -60., (1., -1.), size, &mut self.cloudmap, &self.latitude, &self.heightmap, 0.2, (y - s60) as f64 / (s30 - s60) as f64);
            }
        }
        if s30 != s0 {
            for x in 0..size {
                do_wind(x, s30, s0, -30., 0., (-1., 1.), size, &mut self.cloudmap, &self.latitude, &self.heightmap, 0.5, 1.);
            }
            for y in s30..s0 {
                do_wind(size - 1, y, s0, -30., 0., (-1., 1.), size, &mut self.cloudmap, &self.latitude, &self.heightmap, 0.5, (s0 - y) as f64 / (s0 - s30) as f64);
            }
        }
        if n30 != s0 {
            for x in 0..size {
                do_wind(x, n30, s0, 30., 0., (-1., -1.), size, &mut self.cloudmap, &self.latitude, &self.heightmap, 0.5, 1.);
            }
            for y in s0..n30 {
                do_wind(size - 1, y, s0, 30., 0., (-1., -1.), size, &mut self.cloudmap, &self.latitude, &self.heightmap, 0.5, (y - s0) as f64 / (n30 - s0) as f64);
            }
        }
        if n30 != n60 {
            for x in 0..size {
                do_wind(x, n30, n60, 30., 60., (1., 1.), size, &mut self.cloudmap, &self.latitude, &self.heightmap, 0.2, 1.);
            }
            for y in n30..n60 {
                do_wind(0, y, n60, 30., 60., (1., 1.), size, &mut self.cloudmap, &self.latitude, &self.heightmap, 0.2, (n60 - y) as f64 / (n60 - n30) as f64);
            }
        }

        let max = self.cloudmap.iter().max_by(|x, y| x.partial_cmp(y).unwrap()).unwrap();
        self.cloudmap = self.cloudmap.iter().map(|x| x / max).collect();
    }

    pub fn gen_temp(&mut self) {
        let size = self.noise.size;

        self.tempmap = vec![0.; size * size];

        for i in 0..size * size {
            self.tempmap[i] = clamp(self.insolation[i] * (1. - self.cloudmap[i] / 2.) - (self.heightmap[i] / 4.), 0., 1.);
        }
    }

    pub fn gen_rivermap(&mut self) {
        let size = self.noise.size;

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
                        let neighbs: Vec<(usize, usize)> = self.neighbs[i]
                            .iter()
                            .map(|&(ii, c)| (ii, (10000. * c * (self.heightmap[ii] / (self.heightmap[i] + 0.001))) as usize))
                            .collect();

                        return neighbs;
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

            for (i, &rain) in self.cloudmap.iter().enumerate() {
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

    pub fn gen_watermap(&mut self) {
        let size = self.noise.size;

        self.watermap = vec![0.; size * size];

        for i in 0..size*size {
            if let Some(Water::Lake) = self.waters.get(&i) {
                self.watermap[i] = (self.cloudmap[i] + 1.) / 2.;
            } else if self.heightmap[i] > 0. {
                let best_river = self.neighbs[i]
                    .iter()
                    .map(|&(ii, _)| {
                        if let Some(Water::Lake) = self.waters.get(&ii) {
                            return 1.;
                        } else {
                            return self.rivermap[ii];
                        }
                    })
                    .max_by(|&a, &b| a.partial_cmp(&b).unwrap())
                    .unwrap();
                
                self.watermap[i] = (self.cloudmap[i] + best_river) / (1. + best_river);
            }
        }
    }
    
    pub fn gen_vegetmap(&mut self) {
        let size = self.noise.size;

        self.vegetmap = vec![0.; size * size];

        for i in 0..size*size {
            if self.heightmap[i] > 0. {
                let water = clamp(1.5 * self.watermap[i] - self.tempmap[i] / 2., 0., 1.);

                self.vegetmap[i] = (water * (-(self.tempmap[i] - 0.75).powi(2) + 1.)).sqrt();
            }
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

    pub fn export_minmax<T: Into<PathBuf>>(&self, map: &Vec<f64>, path: T, min: f64, max: f64) {
        let mut i = 0;
        let mut img = RgbImage::new(self.noise.size as u32, self.noise.size as u32);

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
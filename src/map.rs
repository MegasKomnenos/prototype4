use noise::Perlin;
use noise::NoiseFn;
use image::RgbImage;
use image::Rgb;

use std::cmp::max;
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

pub struct ProvBuilder {
    noise: PerlinOctave,
    water_level: f64,
    water_taper: f64,
}

impl ProvBuilder {
    pub fn new(size: usize, freq: f64, pers: f64, lac: f64, min: f64, max: f64, water_level: f64, water_taper: f64,) -> Self {
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
            water_level,
            water_taper,
        }
    }

    pub fn get_image<T: Into<PathBuf>>(&self, path: T) {
        let size = self.noise.size;
        let med = size / 2;

        let mut img = RgbImage::new(size as u32, size as u32);

        for y in 0..size {
            for x in 0..size {
                let dist = ((med as f64 - x as f64).powi(2) + (med as f64 - y as f64).powi(2)).sqrt() / med as f64;
                let val: u8 = match self.noise.get(x as f64, y as f64) - self.water_level - self.water_taper * dist {
                    x if x >= 0. => (x / (1. - self.water_level) * 255.) as u8,
                    _ => 0,
                };

                img.put_pixel(x as u32, y as u32, Rgb([val, val, val]));
            }
        }

        img.save(path.into()).unwrap();
    }
}
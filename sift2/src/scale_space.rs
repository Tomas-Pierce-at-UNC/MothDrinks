
use crate::{
    DELTA_MIN,
    SIGMA_MIN,
    SIGMA_IN,
    SCALES_PER_OCTAVE,
    OCTAVES,
};

const G_BLUR_SQUARE :f64 = (1.0 / DELTA_MIN) * ((SIGMA_MIN * SIGMA_MIN) - (SIGMA_IN * SIGMA_IN));

use crate::image::Image;

use crate::Extremum;

use nalgebra as na;

#[derive(Debug)]
struct Octave {
    scales :[Image; SCALES_PER_OCTAVE + 3],
}

impl Octave {
    fn new_starter(image :&Image) -> Octave {
        let mut scales :Vec<Image> = Vec::with_capacity(SCALES_PER_OCTAVE + 3);
        let initial = image.bilinear_interpolation();
        scales.push(initial);
        for _i in 1..(SCALES_PER_OCTAVE + 3) {
            let last = scales.last().unwrap();
            let blurred = last.gaussian_blur(G_BLUR_SQUARE.sqrt());
            scales.push(blurred);
        }
        let myscales :[Image; SCALES_PER_OCTAVE + 3] = scales.try_into().unwrap();
        Octave {
            scales : myscales
        }
    }

    fn new_following(image :&Image) -> Octave {
        let mut scales :Vec<Image> = Vec::with_capacity(SCALES_PER_OCTAVE + 3);
        let initial = image.downsample();
        scales.push(initial);
        for _i in 1..(SCALES_PER_OCTAVE + 3) {
            let last = scales.last().unwrap();
            let blurred = last.gaussian_blur(G_BLUR_SQUARE.sqrt());
            scales.push(blurred);
        }
        let myscales :[Image; SCALES_PER_OCTAVE + 3] = scales.try_into().unwrap();
        Octave {
            scales : myscales
        }
    }
}

pub struct GaussianScaleSpace {
    octaves :[Octave; OCTAVES],
}

impl GaussianScaleSpace {
    pub fn new(image :&Image) -> GaussianScaleSpace {
        let mut octaves :Vec<Octave> = Vec::with_capacity(OCTAVES);
        let first = Octave::new_starter(&image);
        octaves.push(first);
        for _i in 1..OCTAVES {
            let prev = octaves.last().unwrap();
            let nth = &prev.scales[SCALES_PER_OCTAVE];
            let following = Octave::new_following(nth);
            octaves.push(following);
        }
        let myoctaves :[Octave; OCTAVES] = octaves.try_into().unwrap();

        GaussianScaleSpace { octaves: myoctaves }
    }
}

impl GaussianScaleSpace {

    pub fn gradient(&self, octave :u32, scale :u32, row :u32, col :u32) -> Option<na::Vector2<f64>> {
        let img = self.octaves.get(octave as usize)?.scales.get(scale as usize)?;
        let a = img.get_pixel(row + 1, col)?;
        let b = img.get_pixel(row - 1, col)?;
        let c = img.get_pixel(row, col + 1)?;
        let d = img.get_pixel(row, col - 1)?;

        let partial_m = (a - b) / 2.0;
        let partial_n = (c - d) / 2.0;

        Some(na::vector![partial_m, partial_n])
    }
}

pub struct GradientScaleSpace {
    row_gradient: [[Image; SCALES_PER_OCTAVE + 3]; OCTAVES],
    col_gradient: [[Image; SCALES_PER_OCTAVE + 3]; OCTAVES],
    width :u32,
    height :u32,
}

impl GradientScaleSpace {

    pub fn new(gss :&GaussianScaleSpace) -> GradientScaleSpace {
        let img1 = &gss.octaves[0].scales[0];
        let width = img1.get_width();
        let height = img1.get_height();
        let mut row_grad :[[Image;SCALES_PER_OCTAVE + 3];OCTAVES]= vec![vec![Image::new(width, height);SCALES_PER_OCTAVE + 3].try_into().unwrap(); OCTAVES].try_into().unwrap();
        let mut col_grad = row_grad.clone();

        for (o, octave) in gss.octaves.iter().enumerate() {
            for (s, scale) in octave.scales.iter().enumerate() {
                for row in 0..height {
                    for col in 0..width {
                        if let Some(next_row_val) = scale.get_pixel(row + 1, col) {
                            if let Some(prev_row_val) = scale.get_pixel(row - 1, col) {
                                let r_g = (next_row_val - prev_row_val) / 2.0;
                                match row_grad[o][s].set_pixel(row, col, r_g) {
                                    Some(()) => (),
                                    None => eprintln!("warn: failed to set row gradient"),
                                };
                            }
                        }
                        if let Some(next_col_val) = scale.get_pixel(row, col + 1) {
                            if let Some(prev_col_val) = scale.get_pixel(row, col - 1) {
                                let col_g = (next_col_val - prev_col_val) / 2.0;
                                match col_grad[o][s].set_pixel(row, col, col_g) {
                                    Some(()) => (),
                                    None => eprintln!("warn: failed to set col gradient"),
                                };
                            }
                        }
                    }
                }
            }
        }

        GradientScaleSpace {
            row_gradient : row_grad,
            col_gradient : col_grad,
            width,
            height,
        }
    }

    pub fn get_row_gradient(&self, octave :u32, scale :u32, row :u32, col :u32) -> Option<f64> {
        self.row_gradient.get(octave as usize)?.get(scale as usize)?.get_pixel(row, col)
    }

    pub fn get_col_gradient(&self, octave :u32, scale :u32, row :u32, col :u32) -> Option<f64> {
        self.col_gradient.get(octave as usize)?.get(scale as usize)?.get_pixel(row, col)
    }

    pub fn get_width(&self) -> u32 {
        self.width
    }

    pub fn get_height(&self) -> u32 {
        self.height
    }

}


#[derive(Debug)]
struct OctaveDif {
    scales :[Image; SCALES_PER_OCTAVE + 2],
}

pub struct DoGScaleSpace {
    octaves :[OctaveDif; OCTAVES]
}

pub type DoG = DoGScaleSpace;

impl DoGScaleSpace {

    pub fn new(gauss :&GaussianScaleSpace) -> DoGScaleSpace {
        let mut octaves :Vec<OctaveDif> = Vec::with_capacity(OCTAVES);
        for (_i,octave) in gauss.octaves.iter().enumerate() {
            let mut scales :Vec<Image> = Vec::with_capacity(SCALES_PER_OCTAVE + 2);
            for j in 0..(SCALES_PER_OCTAVE + 2) {
                let here = &octave.scales[j];
                let next = &octave.scales[j + 1];
                let dif = next.difference(&here).unwrap();
                scales.push(dif);
            }
            let myscales = scales.try_into().unwrap();
            let oct = OctaveDif {
                scales : myscales,
            };
            octaves.push(oct);
        }
        let myoctaves = octaves.try_into().unwrap();

        DoGScaleSpace { octaves: myoctaves }
    }

    pub fn build_dog(image :&Image) -> DoGScaleSpace {
        let gauss = GaussianScaleSpace::new(image);
        let dog = DoGScaleSpace::new(&gauss);
        dog
    }
}

impl DoGScaleSpace {

    pub fn get_image(&self, octave :u32, scale :u32) -> Option<&Image> {
        if (octave as usize) < OCTAVES && (scale as usize) < SCALES_PER_OCTAVE + 3 {
            let value = &self.octaves[octave as usize].scales[scale as usize];
            Some(value)
        } else {
            None
        }
    }
}

pub struct Zone {
    center :f64,
    neighbors :[f64; 26],
    valid :usize,
}

impl Zone {

    fn get_valid_neighbors(&self) -> &[f64] {
        &self.neighbors[0..self.valid]
    }

    pub fn new(center :f64) -> Zone {
        Zone {
            center,
            valid : 0,
            neighbors : [0f64; 26],
        }
    }

    pub fn add_neighbor(&mut self, neighbor :f64) -> Option<()> {
        if self.valid >= 26 {
            None
        } else {
            self.neighbors[self.valid] = neighbor;
            self.valid += 1;
            Some(())
        }
    }

    pub fn try_add_neighbor(&mut self, maybe_neighbor :Option<f64>) -> Option<()> {
        match maybe_neighbor {
            Some(n) => self.add_neighbor(n),
            None => None
        }
    }

    /// reports whether the center is a local maxima
    fn is_maxima(&self) -> bool {
        for neighbor in self.get_valid_neighbors() {
            if neighbor > &self.center {
                return false;
            }
        }
        return true;
    }

    /// reports whether the center is a local minima
    fn is_minima(&self) -> bool {
        for neighbor in self.get_valid_neighbors() {
            if neighbor < &self.center {
                return false;
            }
        }
        return true;
    }
}

impl DoGScaleSpace {

    pub fn get_zone(&self, octave :u32, scale :u32, row :u32, col :u32) -> Option<Zone> {
        if (octave as usize) < OCTAVES && (scale as usize) < (SCALES_PER_OCTAVE + 3) {
            let oct = &self.octaves[octave as usize];
            let here_img = &oct.scales[scale as usize];
            if let Some(value) = here_img.get_pixel(row, col) {
                let mut zone = Zone::new(value);
                zone.try_add_neighbor(here_img.get_pixel(row -1, col));
                zone.try_add_neighbor(here_img.get_pixel(row + 1, col));
                zone.try_add_neighbor(here_img.get_pixel(row - 1, col -1));
                zone.try_add_neighbor(here_img.get_pixel(row + 1, col + 1));
                zone.try_add_neighbor(here_img.get_pixel(row, col - 1));
                zone.try_add_neighbor(here_img.get_pixel(row, col + 1));
                zone.try_add_neighbor(here_img.get_pixel(row - 1, col + 1));
                zone.try_add_neighbor(here_img.get_pixel(row + 1, col - 1));
                if let Some(prev) = oct.scales.get((scale - 1) as usize) {
                    for r in (row - 1)..=(row+1) {
                        for c in (col - 1)..=(col + 1) {
                            zone.try_add_neighbor(prev.get_pixel(r,c));
                        }
                    }
                };
                if let Some(next) = oct.scales.get((scale + 1) as usize) {
                    for r in (row-1)..=(row+1) {
                        for c in (col-1)..=(col+1) {
                            zone.try_add_neighbor(next.get_pixel(r,c));
                        }
                    }
                };
                Some(zone)
            } else {
                None
            }
        } else {
            None
        }
    }
}

impl DoGScaleSpace {
    pub fn find_extrema(&self) -> Vec<Extremum> {
        let mut extremes = Vec::new();
        let z = self.octaves[0].scales.get(0).unwrap();
        let width = z.get_width();
        let height = z.get_height();
        for oct in 0..OCTAVES {
            for scale in 0..(SCALES_PER_OCTAVE + 1) {
                for row in 1..(height-1) {
                    for col in 1..(width-1) {
                        if let Some(zone) = self.get_zone(oct as u32, scale as u32, row, col) {
                            if zone.is_maxima() || zone.is_minima() {
                                let o = oct as u32;
                                let s = scale as u32;
                                let extreme = Extremum {
                                    octave : o,
                                    scale :s,
                                    row,
                                    col,
                                    value : zone.center
                                };
                                extremes.push(extreme);
                            }
                        }
                    }
                }
            }
        }
        extremes
    }
}
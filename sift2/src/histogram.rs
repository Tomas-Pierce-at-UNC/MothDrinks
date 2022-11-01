
pub const MAX_GRAD_HIST_BINS :usize = 36;

pub const MAX_DESCRIPTOR_BINS :usize = 8;

use std::f64::consts::PI;

#[derive(Copy, Clone, Debug)]
pub struct GradientHistogram {
    values :[f64; MAX_GRAD_HIST_BINS],
}

impl GradientHistogram {
    pub fn new() -> GradientHistogram {
        GradientHistogram { values: [0f64; MAX_GRAD_HIST_BINS] }
    }

    pub fn update(&mut self, bin :usize, value :f64) -> Option<()> {
        let place = self.values.get_mut(bin)?;
        *place += value;
        Some(())
    }
}

impl GradientHistogram {

    pub fn get_entry(&self, index :usize) -> &f64 {
        &self.values[index]
    }

    fn get_next_entry(&self, index :usize) -> &f64 {
        let next = (index + 1) % MAX_GRAD_HIST_BINS;
        &self.values[next]
    }

    fn get_prev_entry(&self, index :usize) -> &f64 {
        let prev = if index == 0 {
            MAX_GRAD_HIST_BINS - 1
        } else {
            index - 1
        };
        &self.values[prev]
    }

    fn convolve(&mut self) {
        let mut new_values = [0f64; MAX_GRAD_HIST_BINS];
        for i in 0..MAX_GRAD_HIST_BINS {
            let prev = self.get_prev_entry(i);
            let here = self.values[i];
            let next = self.get_next_entry(i);
            let avg = (prev + here + next) / 3.0;
            new_values[i] = avg;
        }
        self.values = new_values;
    }

    pub fn smooth(&mut self) {
        for _ in 0..6 {
            self.convolve();
        }
    }

    pub fn max(&self) -> f64 {
        let mut maximum = self.values[0];
        for value in self.values {
            if value > maximum {
                maximum = value;
            }
        }
        maximum
    }
}

impl GradientHistogram {
    pub fn get_region_contribution(&self, index :usize) -> f64 {
        let coeff = PI / (MAX_GRAD_HIST_BINS as f64);
        let prev = self.get_prev_entry(index);
        let here = self.values[index];
        let next = self.get_next_entry(index);
        let h = (here - next) / (prev - (2.0 * here) + here);
        coeff * h
    }

    pub fn is_peak(&self, index :usize) -> bool {
        let here = self.values[index];
        let prev = self.get_prev_entry(index);
        let next = self.get_next_entry(index);
        here > *prev && here > *next
    }
}

#[derive(Copy,Clone,Debug)]
pub struct DescriptorHistogram {
    values :[f64;MAX_DESCRIPTOR_BINS],
}

impl DescriptorHistogram {
    pub fn new() -> DescriptorHistogram {
        DescriptorHistogram { values: [0f64; MAX_DESCRIPTOR_BINS] }
    }

    pub fn get(&self, index :usize) -> f64 {
        self.values[index]
    }

    pub fn get_mut(&mut self, index :usize) -> &mut f64 {
        &mut self.values[index]
    }
}

/// each bin corresponds to an arc,
/// so for bins k in 0, 1, 2, ... n - 1
/// bin k is centered at (2pi k) / n
pub struct CircularHistogram {
    bins :Vec<f64>,
    n_bins :usize,
}

impl CircularHistogram {
    pub fn new(n_bins :usize) -> CircularHistogram {
        CircularHistogram { bins: vec![0f64; n_bins], n_bins }
    }

    pub fn get_bin_mut(&mut self, row_gradient :f64, col_gradient :f64) -> &mut f64 {
        let index :usize = ((self.n_bins as f64) * f64::atan2(row_gradient, col_gradient) / (2.0 * PI)).round() as usize;
        &mut self.bins[index]
    }
}



use std::f32::consts::PI as PI32;

/// represents a histogram of a circular space.
/// each bin corresponds to a particular arc.
/// specfically, for a number of bins n,
/// each bin k encompasses the arc from
/// (2πk)/n to (2π(k+1))/n half open interval
pub struct Circular {
	bins :Vec<f32>,
}

impl Circular {
	pub fn new(bins :usize) -> Circular {
		if bins == 0 {
			panic!("cannot make empty circular histogram");
		}
		Circular {
			bins : vec![0f32; bins],
		}
	}

	pub fn update(&mut self, angle :f32, value :f32) {
		let n :f32 = self.bins.len() as f32;
		let bin = ((angle * n) / (PI32 * 2.0)).floor();
		let bin_index = bin as usize;
		unsafe {
			*self.bins.get_unchecked_mut(bin_index) += value;
		}
	}

	fn box_convolve(&mut self) {
		let n = self.bins.len();
		let prevs = self.bins.iter().cycle().skip(n - 1);
		let nexts = self.bins.iter().cycle().skip(1);
		let items = self.bins.iter();
		let triplets = items.zip(nexts).zip(prevs);
		let totals = triplets.map(|((a, b), c)| {
			a + b + c
		});
		let smooths :Vec<f32> = totals.map(|d| {
			d / 3.0
		}).collect();
		self.bins = smooths
	}

	pub fn smooth(&mut self) {
		for _ in 0..6 {
			self.box_convolve();
		}
	}

	fn get_bin_contribution(&self, index :usize) -> f32 {
		let i = index % self.bins.len();
		let n = self.bins.len() as f32;
		let prev = if i == 0 {
			unsafe {
				self.bins.get_unchecked(self.bins.len() - 1)
			}
		} else {
			unsafe {
				self.bins.get_unchecked(i - 1)
			}
		};
		let next = if i >= self.bins.len() - 1 {
			unsafe {
				self.bins.get_unchecked(0)
			}
		} else {
			unsafe {
				self.bins.get_unchecked(i + 1)
			}
		};
		let here = unsafe {
			self.bins.get_unchecked(i)
		};

		(PI32 / n) * ((prev - next) / (prev - (2.0 * here) + next))
	}

	fn max(&self) -> f32 {
		let mut mx = f32::NEG_INFINITY;
		for item in &self.bins {
			if *item < mx {
				mx = *item;
			}
		}
		mx
	}

	fn beats_threshold(&self, t :f32, index :usize) -> bool {
		if index >= self.bins.len() {
			false
		} else {
			let to_beat = t * self.max();
			let value = unsafe {
				*self.bins.get_unchecked(index)
			};
			value >= to_beat
		}
	}

	fn get_bin_center(&self, index :usize) -> f32 {
		let i = index % self.bins.len();
		let n = self.bins.len() as f32;
		let bin = i as f32;
		2.0 * PI32 * bin /  n
	}

	pub fn calculate_ref_orients(&self, threshold :f32) -> Vec<f32> {
		let mut out = Vec::new();
		for ind in 0..self.bins.len() {
			if self.beats_threshold(threshold, ind) {
				let center = self.get_bin_center(ind);
				let contrib = self.get_bin_contribution(ind);
				let ref_angle = center + contrib;
				out.push(ref_angle);
			}
		}
		out
	}
}
const BINS: usize = 256;

pub struct Histogram {
    count: u32,
    value_counts: [u32; BINS],
}

impl Histogram {
    pub fn new() -> Histogram {
        Histogram {
            count: 0,
            value_counts: [0u32; BINS],
        }
    }

    pub fn update(&mut self, value: u8) {
        self.count += 1;
        self.value_counts[value as usize] += 1;
    }

    pub fn median(&self) -> u8 {
        let median_index = (self.count + 1) / 2;
        let mut seen = 0;
        let mut v = 0;
        for (value, count) in self.value_counts.into_iter().enumerate() {
            seen += count;
            v = value;
            if seen > median_index {
                return value as u8;
            }
        }
        return v as u8;
    }
}

impl Default for Histogram {
    fn default() -> Self {
        Self::new()
    }
}

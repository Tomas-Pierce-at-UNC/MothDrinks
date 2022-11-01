
use convolve2d::Matrix;
use convolve2d::MatrixMut;
use convolve2d::write_convolution;
use convolve2d::kernel;

#[derive(Clone)]
#[derive(Debug)]
pub struct Image {
    data :Vec<f64>,
    width :u32,
    height :u32,
}

impl Image {

    pub fn new(width :u32, height :u32) -> Image {
        let data = vec![0f64;(width * height) as usize];
        Image {
            data,
            width,
            height
        }
    }

    /// assumes that the image data has been aranged in 'C'
    /// or row-major order
    pub unsafe fn load(&mut self, data :*const f64) {
        let len = self.data.len();
        let dat = self.data.as_mut_ptr();
        data.copy_to(dat, len);
    }

    pub unsafe fn load_bytes(&mut self, bytes :*const u8) {
        let len = self.data.len();
        for item in 0..len {
            let byte_value = *bytes.add(item);
            let float_val = byte_value as f64;
            let normalized = float_val / 255.0;
            self.data[item] = normalized;
        }
    }


    pub fn get_pixel(&self, row :u32, col :u32) -> Option<f64> {
        if row >= self.height || col >= self.width {
            None
        } else {
            let offset = self.find_offset(row, col);
            let value = unsafe {
                *self.data.get_unchecked(offset)
            };
            Some(value) 
        }
    }

    fn find_offset(&self, row :u32, col :u32) -> usize {
        ((row * self.width) + col) as usize
    }

    pub fn set_pixel(&mut self, row :u32, col :u32, value :f64) -> Option<()> {
        if row >= self.height || col >= self.width {
            None
        } else {
            let offset = self.find_offset(row, col);
            let myref = unsafe {
                self.data.get_unchecked_mut(offset)
            };
            *myref = value;
            Some(())
        }
    }

    pub fn get_width(&self) -> u32 {
        self.width
    }

    pub fn get_height(&self) -> u32 {
        self.height
    }
}

/// assumes imagedata is in 'C' or row-major order
pub fn build_image(width :u32, height :u32, data :*const f64) -> Option<Image> {
    let mut image = Image::new(width, height);
    if data.is_null() {
        None
    } else {
        unsafe {
            image.load(data);
        }
        Some(image)
    }
}

impl Matrix<f64> for Image {

    fn get_width(&self) -> usize {
        self.width as usize
    }

    fn get_height(&self) -> usize {
        self.height as usize
    }

    fn get_value(&self, row :usize, col :usize) -> Option<&f64> {
        if row >= self.height as usize || col >= self.width as usize {
            None
        } else {
            let offset :usize = ((row * self.width as usize) + col) as usize;
            self.data.get(offset)
        }
    }

    fn get_data(&self) -> &[f64] {
        self.data.as_slice()
    }
}

impl MatrixMut<f64> for Image {

    fn get_data_mut(&mut self) -> &mut [f64] {
        self.data.as_mut_slice()
    }

}


impl Image {

    /// takes the gaussian blur of the image with a sigma of blur_level.
    pub fn gaussian_blur(&self, blur_level :f64) -> Image {
        let g_kern = kernel::gaussian(3, blur_level);
        let mut out = Image::new(self.width, self.height);
        write_convolution(self, &g_kern, &mut out);
        out
    }

    /// A times 2 bilinear interpolation, corresponds to a delta_min = 0.5.
    /// used in the construction of scale space.
    pub fn bilinear_interpolation(&self) -> Image {

        let mut out = Image::new(self.width * 2, self.height * 2);
        for row in 0..self.height {
            for col in 0..self.width {
                let here = self.get_pixel(row, col).unwrap();

                out.set_pixel(row * 2, col * 2, here).unwrap();

                if let Some(pix) = self.get_pixel(row, col + 1) {
                    let x1 = col as f64;
                    let x2 = (col + 1) as f64;
                    let y1 = here as f64;
                    let y2 = pix as f64;
                    let slope = (y2 - y1) / (x2 - x1);
                    let mid_x = (x1 + x2) / 2.0;
                    let interpolated_value = (slope * (mid_x - x1)) + y1;
                    match out.set_pixel(row * 2, (col * 2) + 1, interpolated_value) {
                        Some(()) => (),
                        None => eprintln!("warning: attempt to write into non-existent pixel of image"),
                    };
                } else {
                    match out.set_pixel(row * 2, (col * 2) + 1, here) {
                        Some(()) => (),
                        None => eprintln!("warning: attempt to write into non-existent pixel of image"),
                    };
                }
            }
        }
        for row in 0..self.height {
            for col in 0..self.width {
                let here = self.get_pixel(row, col).unwrap();
                if let Some(pix) = self.get_pixel(row + 1, col) {
                    let x1 = row as f64;
                    let x2 = (row + 1) as f64;
                    let y1 = here as f64;
                    let y2 = pix as f64;
                    let slope = (y2 - y1) / (x2 - x1);
                    let mid_x = (x1 + x2) / 2.0;
                    let interpolated_value = slope * (mid_x - x1) + y1;
                    match out.set_pixel((row * 2) + 1, col * 2, interpolated_value) {
                        Some(()) => (),
                        None => eprintln!("warning; attempt to write into non-existent pixel of image"),
                    };
                } else {
                    match out.set_pixel((row * 2) + 1, col * 2, here) {
                        Some(()) => (),
                        None => eprintln!("warning: attempt to write into non-exi")
                    }
                }
            }
        }

        out
    }

    /// downsamples the image by a factor of two in both dimensions.
    /// does not actually change the size of the image.
    /// instead causes the information density to halve in both dimensions.
    pub fn downsample(&self) -> Image {
        let mut out = Image::new(self.width, self.height);
        for row in 0..self.height {
            for col in 0..self.width {
                if row % 2 == 0 && col % 2 == 0 {
                    let value = self.get_pixel(row, col).unwrap();
                    out.set_pixel(row, col, value);
                } else if row % 2 == 0 && col % 2 != 0 {
                    let value = self.get_pixel(row, col - 1).unwrap();
                    out.set_pixel(row, col, value);
                } else if row % 2 != 0 && col % 2 == 0 {
                    let value = self.get_pixel(row - 1, col).unwrap();
                    out.set_pixel(row, col, value);
                } else { /* row % 2 != 0 && col % 2 != 0 */
                    let value = self.get_pixel(row - 1, col - 1).unwrap();
                    out.set_pixel(row, col, value);
                }
            }
        }
        out
    }
}

impl Image {
    pub fn difference(&self, other :&Image) -> Option<Image> {
        if self.width == other.width && self.height == other.height {
            let deltas = self.data.iter().zip(other.data.iter()).map(|(a,b)| {
                a - b
            }).collect();
            let out :Image = Image {
                width : self.width,
                height : self.height,
                data : deltas
            };
            Some(out)
        } else {
            None
        }
    }
}

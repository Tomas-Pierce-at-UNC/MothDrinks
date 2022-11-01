
use crate::DELTA_MIN;
use crate::Extremum;
use crate::C_DOG;
use crate::SCALES_PER_OCTAVE;
use crate::SIGMA_MIN;
use crate::get_delta_o;
use crate::C_EDGE;
use crate::scale_space;

use crate::histogram::GradientHistogram;
use crate::histogram::MAX_GRAD_HIST_BINS;
use crate::histogram as hist;

use crate::scale_space::DoG;

use nalgebra as na;
use rayon::prelude::*;

use ordered_float::NotNan;

pub fn conservative_discard_low_contrast(extrema :&mut Vec<Extremum>) {
    extrema.retain(|extreme :&Extremum| -> bool {
        extreme.value.abs() >= 0.8 * C_DOG
    });

}

pub fn discard_low_contrast(canidates :&mut Vec<Keypoint>) {
    canidates.retain(|canidate :&Keypoint| {
        canidate.omega >= C_DOG
    });
}

fn get_2d_hessian(keypoint :&Keypoint, dog :&DoG) -> Option<na::Matrix2<f64>> {
    let double = keypoint.val * 2.0;
    let row = keypoint.row;
    let col = keypoint.col;
    let scale = keypoint.scale;
    let octave = keypoint.octave;

    let here = dog.get_image(octave, scale)?;
    let a = here.get_pixel(row + 1, col)?;
    let b = here.get_pixel(row - 1, col)?;

    let h11 = a + b - double;

    let c = here.get_pixel(row, col + 1)?;
    let d = here.get_pixel(row, col - 1)?;

    let h22 = c + d - double;

    let e = here.get_pixel(row + 1, col + 1)?;
    let f = here.get_pixel(row + 1, col - 1)?;
    let g = here.get_pixel(row - 1, col + 1)?;
    let h = here.get_pixel(row - 1, col - 1)?;

    let h12 = (e - f - g + h) / 4.0;

    Some(na::matrix![h11, h12; h12, h22])
}

pub fn discard_on_edges(canidates :&mut Vec<Keypoint>, dog :&DoG) {
    canidates.retain(|key :&Keypoint| {
        if let Some(hessian) = get_2d_hessian(key, dog) {
            let edgeness = hessian.trace().powi(2) / hessian.determinant();
            edgeness < (C_EDGE + 1.0).powi(2) / C_EDGE
        } else {
            false
        }
    })
}

#[derive(Clone)]
pub struct Keypoint {
    octave :u32,
    scale :u32,
    row :u32,
    col :u32,
    x :f64,
    y :f64,
    sigma :f64,
    omega :f64,
    val :f64,
}

pub fn interpolate_keypoints(extrema :&[Extremum], dog :&DoG) -> Vec<Keypoint> {
    let mut out = Vec::new();
    for extreme in extrema.iter() {
        let mut scale = extreme.scale;
        let mut row = extreme.row;
        let mut col = extreme.col;
        let delta_o = get_delta_o(extreme.octave as i32);
        for _ in 0..5 {
            if let Some((a_star, omega)) = quadratic_interpolation(extreme, dog) {
                let sigma = calculate_sigma(a_star[0], delta_o, scale as f64);
                let x = delta_o * (a_star[1] + row as f64);
                let y = delta_o * (a_star[2] + col as f64);
                scale = (scale as f64 + a_star[0]).round() as u32;
                row = (row as f64 + a_star[1]).round() as u32;
                col = (col as f64 + a_star[2]).round() as u32;
                let max = maximum(a_star[0].abs(), maximum(a_star[1].abs(), a_star[2].abs()));
                if max < 0.6 {
                    let keypoint = Keypoint {
                        octave : extreme.octave,
                        scale,
                        row,
                        col,
                        x,
                        y,
                        sigma,
                        omega,
                        val: extreme.value,
                    };
                    out.push(keypoint);
                }
            } else {
                break;
            }
        }
    }
    out
}

fn calculate_sigma(a1 :f64, delta_o :f64, scale :f64) -> f64 {
    (delta_o / DELTA_MIN) * SIGMA_MIN * 2f64.powf((a1 + scale) / SCALES_PER_OCTAVE as f64)
}

fn maximum(left :f64, right :f64) -> f64 {
    if left >= right {
        left
    } else {
        right
    }
}


pub fn quadratic_interpolation(extremum :&Extremum, dog :&DoG) -> Option<(na::Vector3<f64>, f64)> {
    let octave = extremum.octave;
    let here_scale = extremum.scale;
    let here_row = extremum.row;
    let here_col = extremum.col;

    let here_val = extremum.value;

    let next_scale = dog.get_image(octave, here_scale + 1)?;
    let next_scale_val = next_scale.get_pixel(here_row, here_col)?;
    let prev_scale = dog.get_image(octave, here_scale - 1)?;
    let prev_scale_val = prev_scale.get_pixel(here_row, here_col)?;

    let here = dog.get_image(octave, here_scale)?;

    let next_row_val = here.get_pixel(here_row + 1, here_col)?;
    let prev_row_val = here.get_pixel(here_row - 1, here_col)?;

    let next_col_val = here.get_pixel(here_row, here_col + 1)?;
    let prev_col_val = here.get_pixel(here_row, here_col - 1)?;

    let gradient_3d = na::vector![(next_scale_val - prev_scale_val) / 2.0, 
    (next_row_val - prev_row_val) / 2.0,
    (next_col_val - prev_col_val) / 2.0
    ];

    let double = 2.0 * here_val;

    let h11 = next_scale_val + prev_scale_val - double;
    let h22 = next_row_val + prev_row_val - double;
    let h33 = next_col_val + prev_col_val - double;

    let g = next_scale.get_pixel(here_row + 1, here_col)?;
    let h = next_scale.get_pixel(here_row - 1, here_col)?;
    let i = prev_scale.get_pixel(here_row + 1, here_col)?;
    let j = prev_scale.get_pixel(here_row - 1, here_col)?;

    let h12 = (g - h - i + j) / 4.0;

    let k = next_scale.get_pixel(here_row, here_col + 1)?;
    let l = next_scale.get_pixel(here_row, here_col - 1)?;
    let m = prev_scale.get_pixel(here_row, here_col + 1)?;
    let n = prev_scale.get_pixel(here_row, here_col - 1)?;

    let h13 = (k - l - m + n) / 4.0;

    let o = here.get_pixel(here_row + 1, here_col + 1)?;
    let p = here.get_pixel(here_row + 1, here_col - 1)?;
    let q = here.get_pixel(here_row - 1, here_col + 1)?;
    let r = here.get_pixel(here_row - 1, here_col - 1)?;

    let h23 = (o - p - q + r) / 4.0;

    let hessian_3d = na::matrix![
        h11, h12, h13;
        h12, h22, h23;
        h13, h22, h33
    ];

    let decomposed_hessian = na::LU::new(hessian_3d);
    let neg_a_star = decomposed_hessian.solve(&gradient_3d)?;
    let a_star = - neg_a_star;

    let omega = here_val - (0.5 * gradient_3d.dot(&neg_a_star));

    Some((a_star, omega))
}

#[derive(Copy, Clone)]
pub struct OrientedKeypoint {
    octave :u32,
    scale :u32,
    x :f64,
    y :f64,
    sigma :f64,
    orientation :f64,
}

const TWO_PI :f64 = std::f64::consts::PI * 2.0;

pub fn orient_keypoints(keypoints :&[Keypoint], gradient :&scale_space::GradientScaleSpace, lambda_ori :f64, t :f64) -> Vec<OrientedKeypoint> {
    let h = gradient.get_height() as f64;
    let w = gradient.get_width() as f64;
    let mut out = Vec::new();
    'keys: for key_pt in keypoints {
        let triple = 3.0 * lambda_ori * key_pt.sigma;
        if triple <= key_pt.x && key_pt.x <= h - triple && triple <= key_pt.y && key_pt.y <= w - triple {
            let mut hist = GradientHistogram::new();
            let delta_o = DELTA_MIN * (2f64).powi(key_pt.octave as i32 - 1);
            let row_min = ((key_pt.x - triple) / delta_o).round() as u32;
            let row_max = ((key_pt.x + triple) / delta_o).round() as u32;
            let col_min = ((key_pt.y - triple) / delta_o).round() as u32;
            let col_max = ((key_pt.y + triple) / delta_o).round() as u32;
            for row in row_min..row_max + 1 {
                for col in col_min..col_max + 1 {
                    let row_grad = match gradient.get_row_gradient(key_pt.octave, key_pt.scale, row, col) {
                        Some(g) => g,
                        None => continue 'keys,
                    };
                    let col_grad = match gradient.get_col_gradient(key_pt.octave, key_pt.scale, row, col) {
                        Some(g) => g,
                        None => continue 'keys,
                    };
                    let grad_mag = (row_grad.powi(2) + col_grad.powi(2)).sqrt();
                    let m = row as f64;
                    let n = col as f64;
                    let delta_mag_sq = ((m * delta_o) - key_pt.x).powi(2) + ((n * delta_o) - key_pt.y).powi(2);
                    let coeff = f64::exp(-delta_mag_sq / (2.0 * (lambda_ori * key_pt.sigma).powi(2)));
                    let contrib = coeff * grad_mag;
                    let angle = f64::atan2(row_grad, col_grad) % TWO_PI;
                    let bins = MAX_GRAD_HIST_BINS as f64;
                    let bin = ((bins / TWO_PI) * angle).round() as u32;
                    hist.update(bin as usize, contrib);
                }
            }

            hist.smooth();

            for i in 0..MAX_GRAD_HIST_BINS {
                if hist.is_peak(i) && *hist.get_entry(i) >= (t * hist.max()) {
                    let theta_k = TWO_PI * (i as f64) / (MAX_GRAD_HIST_BINS as f64);
                    let theta_key = theta_k + hist.get_region_contribution(i);
                    let oriented = OrientedKeypoint {
                        octave : key_pt.octave,
                        scale : key_pt.scale,
                        x : key_pt.x,
                        y : key_pt.y,
                        sigma : key_pt.sigma,
                        orientation : theta_key,
                    };
                    out.push(oriented);
                }
            }
        }
    }
    out
}

#[derive(Clone)]
pub struct Descriptor {
    pub keypoint :OrientedKeypoint,
    pub features :Vec<f64>,
}


impl Descriptor {
    pub fn get_x(&self) -> f64 {
        self.keypoint.x
    }

    pub fn get_y(&self) -> f64 {
        self.keypoint.y
    }

    pub fn distance_squared(&self, other :&Descriptor) -> NotNan<f64> {
        let key = self.keypoint;
        let okey = other.keypoint;
        let mut out :f64 = 0.0;
        for item in self.features.iter().zip(other.features.iter()) {
            let part = (item.1 - item.0).powi(2);
            out += part;
        }
        // out += ((key.octave - okey.octave) as f64).powi(2);
        // out += ((key.scale - okey.scale) as f64).powi(2);
        // out += ((key.row - okey.row) as f64).powi(2);
        // out += ((key.col - okey.col) as f64).powi(2);
        out += (key.x - okey.x).powi(2);
        out += (key.y - okey.y).powi(2);
        out += (key.sigma - okey.sigma).powi(2);
        // out += (key.omega - okey.omega).powi(2);
        // out += (key.val - okey.val).powi(2);
        out += (key.orientation - okey.orientation).powi(2);
        NotNan::new(out).unwrap()
    }
}

use std::f64::consts::SQRT_2;

pub fn construct_descriptors(keypoints :&[OrientedKeypoint], gradient :&scale_space::GradientScaleSpace, n_hist :u32, n_ori :u32, lambda_desc :f64) -> Vec<Descriptor> {
    let mut out = Vec::new();
    
    let w = gradient.get_width() as f64;
    let h = gradient.get_height() as f64;
    'keys: for key_pt in keypoints {
        let radius = SQRT_2 * lambda_desc * key_pt.sigma;
        if radius <= key_pt.x && key_pt.x <= h - radius && radius <= key_pt.y && key_pt.y <= w - radius {
            let mut histograms = vec![hist::DescriptorHistogram::new();(n_ori * n_ori) as usize];
            let nhist_ratio = ((n_hist as f64) + 1.0) / (n_hist as f64);
            let delta_o = DELTA_MIN * 2f64.powi(key_pt.octave as i32 - 1);
            let discrep = radius * nhist_ratio;
            let low_m  = ((key_pt.x - discrep) / delta_o).round() as u32;
            let high_m = ((key_pt.x + discrep) / delta_o).round() as u32;
            let low_n = ((key_pt.y - discrep) / delta_o).round() as u32;
            let high_n = ((key_pt.y + discrep) / delta_o).round() as u32;
            for m in low_m..high_m + 1 {
                for n in low_n..high_n + 1 {
                    let row = m as f64;
                    let col = n as f64;
                    let xnorm = ((((row * delta_o) - key_pt.x)*key_pt.orientation.cos()) + (((col*delta_o) - key_pt.y)*key_pt.orientation.sin())) / key_pt.sigma;
                    let ynorm = (((-(row*delta_o) - key_pt.x)*key_pt.orientation.sin()) + (((col * delta_o) - key_pt.y)*key_pt.orientation.cos())) / key_pt.sigma;
                    if xnorm.abs() < (lambda_desc * nhist_ratio) && ynorm.abs() < (lambda_desc * nhist_ratio) {
                        let rgrad = match gradient.get_row_gradient(key_pt.octave, key_pt.scale, m, n) {
                            Some(r) => r,
                            None => continue 'keys,
                        };
                        let cgrad = match gradient.get_col_gradient(key_pt.octave, key_pt.scale, m, n) {
                            Some(c) => c,
                            None => continue 'keys,
                        };
                        let theta_norm = (f64::atan2(rgrad, cgrad) - key_pt.orientation) % TWO_PI;

                        let grad_mag = (rgrad.powi(2) + cgrad.powi(2)).sqrt();

                        let exp = (((row*delta_o) - key_pt.x).powi(2) + ((col*delta_o) - key_pt.y).powi(2)) / (2.0 * (lambda_desc*key_pt.sigma).powi(2));

                        let contrib = f64::exp(-exp) * grad_mag;

                        'xi: for i in 1..=n_hist {
                            'yj: for j in 1..=n_hist {
                                let x_i = (i as f64 - (1.0 + n_hist as f64) / 2.0) * (2.0 * lambda_desc / n_hist as f64);
                                let x_i_0 = x_i - 1.0;
                                let y_j = (j as f64 - (1.0 + n_hist as f64) / 2.0) * (2.0 * lambda_desc / n_hist as f64);
                                let y_j_0 = y_j - 1.0;
                                let ratio = (2.0 * lambda_desc) / (n_hist as f64);
                                if (x_i_0 - xnorm).abs() <= ratio {
                                    if (y_j_0 - ynorm).abs() <= ratio {
                                        let hist = &mut histograms[(i as usize * n_ori as usize) + j as usize];
                                        for k in 0..n_ori {
                                            let theta_k = (TWO_PI*(k as f64 - 1.0)) / n_ori as f64;
                                            let value = hist.get_mut(k as usize);
                                            let ration = n_hist as f64 / (2.0 *lambda_desc);
                                            let xcom = 1.0 - (ration*(xnorm - x_i_0).abs());
                                            let ycom = 1.0 - (ration*(ynorm - y_j_0).abs());
                                            let thetacom = 1.0 - ((n_ori as f64 / TWO_PI)*((theta_norm - theta_k) % TWO_PI).abs());
                                            *value += xcom*ycom*thetacom*contrib;
                                        }
                                    } else {
                                        continue 'yj;
                                    }
                                } else {
                                    continue 'xi;
                                }
                            }
                        }
                    }
                }
            }

            let mut feature = Vec::new();

            for i in 0..n_hist {
                for j in 0..n_hist {
                    let hist = &histograms[(i as usize * n_ori as usize) + j as usize];
                    for k in 0..n_ori {
                        let val = hist.get(k as usize);
                        feature.push(val);
                    }
                }
            }

            let p_feature = feature.par_iter();
            let squares = p_feature.map(|a :&f64| -> f64 { a.powi(2)});
            let total :f64 = squares.sum();
            let norm :f64 = total.sqrt();
            for i in 0..feature.len() {
                if (0.2 * norm) < feature[i] {
                    feature[i] = 0.2 * norm;
                }
                let alternate = (512.0 * feature[i]) / norm;
                if alternate < 255.0 {
                    feature[i] = alternate;
                } else {
                    feature[i] = 255.0;
                }
            }

            let descriptor = Descriptor {
                keypoint : *key_pt,
                features : feature,
            };
            out.push(descriptor);
        }
    }
    out
}

use crate::N_HIST;
use crate::N_ORI;

#[repr(C)]
pub struct DescribedKeypoint {
    octave :u32,
    scale :u32,
    x :f64,
    y :f64,
    sigma :f64,
    orientation :f64,
    features :[f64; (N_HIST * N_HIST * N_ORI) as usize],
}

impl From<Descriptor> for DescribedKeypoint {
    fn from(descriptor :Descriptor) -> DescribedKeypoint {
        let mut out = DescribedKeypoint {
            octave : descriptor.keypoint.octave,
            scale : descriptor.keypoint.scale,
            x : descriptor.keypoint.x,
            y : descriptor.keypoint.y,
            sigma : descriptor.keypoint.sigma,
            orientation : descriptor.keypoint.orientation,
            features : [0f64; (N_HIST * N_HIST * N_ORI) as usize],
        };
        for i in 0..(N_HIST * N_HIST * N_ORI) {
            match descriptor.features.get(i as usize) {
                Some(v) => {out.features[i as usize] = *v;}
                None => break,
            }
        }
        out
    }
}
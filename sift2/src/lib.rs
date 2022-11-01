

pub mod histogram;
pub mod image;
pub mod keypoint;
pub mod scale_space;
pub mod quadtree;

use image::Image;

use std::mem;

#[derive(Clone)]
pub struct Extremum {
    pub octave :u32,
    pub scale :u32,
    pub row :u32,
    pub col :u32,
    pub value :f64,
}

pub use keypoint::DescribedKeypoint;

const OCTAVES :usize = 8;
const SCALES_PER_OCTAVE :usize = 3;
const SIGMA_MIN :f64 = 0.8;
const DELTA_MIN :f64 = 0.5;
const SIGMA_IN :f64 = 0.5;
const C_DOG :f64 = 0.015;
const C_EDGE :f64 = 10.0;
const C_MATCH_REL :f64 = 0.6;

const N_HIST :u32 = 4;
const N_ORI :u32 = 8;
const LAMBDA_DESC :f64 = 6.0;
const PARAM_T :f64 = 0.8;
const LAMBDA_ORI :f64 = 1.5;

pub fn get_delta_o(octave :i32) -> f64 {
    DELTA_MIN * (2.0f64).powi(octave - 1)
}

pub fn find_keypoints(image :&Image) -> Vec<keypoint::Keypoint> {
    let gauss = scale_space::GaussianScaleSpace::new(image);
    let dog = scale_space::DoGScaleSpace::new(&gauss);
    let mut extremes = dog.find_extrema();
    keypoint::conservative_discard_low_contrast(&mut extremes);
    let mut interpolated = keypoint::interpolate_keypoints(&extremes, &dog);
    keypoint::discard_low_contrast(&mut interpolated);
    keypoint::discard_on_edges(&mut interpolated, &dog);
    interpolated
}

pub fn find_descriptors(image :&Image) -> Vec<keypoint::Descriptor> {
    let gauss = scale_space::GaussianScaleSpace::new(image);
    let dog = scale_space::DoGScaleSpace::new(&gauss);
    let mut extremes = dog.find_extrema();
    keypoint::conservative_discard_low_contrast(&mut extremes);
    let mut interpolated = keypoint::interpolate_keypoints(&extremes, &dog);
    keypoint::discard_low_contrast(&mut interpolated);
    keypoint::discard_on_edges(&mut interpolated, &dog);
    let gradient = scale_space::GradientScaleSpace::new(&gauss);
    let oriented = keypoint::orient_keypoints(&interpolated, &gradient, LAMBDA_ORI, PARAM_T);
    let descriptors = keypoint::construct_descriptors(&oriented, &gradient, N_HIST, N_ORI, LAMBDA_DESC);
    descriptors
}

pub fn find_matches(left :Vec<keypoint::Descriptor>, right :Vec<keypoint::Descriptor>) -> Vec<(keypoint::Descriptor, keypoint::Descriptor)> {
    let mut out = Vec::new();
    let r_tree = quadtree::Node::build(right);
    for leftist in left {
        let mut closers = r_tree.find_closest_approx_multiple(&leftist);
        let first_closest = match closers.pop() {
            Some(c) => c,
            None => continue,
        };
        let second_closest = match closers.pop() {
            Some(c) => c,
            None => continue,
        };
        let dist1 = leftist.distance_squared(&first_closest).sqrt();
        let dist2 = leftist.distance_squared(&second_closest).sqrt();
        if dist1 < C_MATCH_REL * dist2 {
            out.push((leftist, first_closest.clone()));
        }
    }
    out
}

#[no_mangle]
pub unsafe extern "C" fn find_matching_descriptors(width :u32, height :u32, img_ref_data :*const f64, img_target_data:*const f64) -> *const DescribedKeypoint {
    let ref_img = match image::build_image(width, height, img_ref_data) {
        Some(img) => img,
        None => {return std::ptr::null();}
    };
    let target_img = match image::build_image(width, height, img_target_data) {
        Some(img) => img,
        None => {return std::ptr::null();}
    };
    let ref_descriptors = find_descriptors(&ref_img);
    let target_descriptors = find_descriptors(&target_img);
    let matches = find_matches(ref_descriptors, target_descriptors);
    let mut out :Vec<DescribedKeypoint> = Vec::with_capacity(matches.len() * 2);
    for (ref_desc, target_desc) in matches {
        let a = ref_desc.into();
        out.push(a);
        let b = target_desc.into();
        out.push(b);
    }
    let ptr = out.as_ptr();
    mem::forget(out);
    ptr
}

#[repr(C)]
pub struct KeyVector {
    count :usize,
    keys :*const DescribedKeypoint,
}

#[no_mangle]
pub unsafe extern "C" fn find_image_matches(w1 :u32, h1 :u32, w2 :u32, h2 :u32, img1_data :*const u8, img2_data :*const u8) -> KeyVector {
    let mut img1 = Image::new(w1, h1);
    let mut img2 = Image::new(w2, h2);
    img1.load_bytes(img1_data);
    img2.load_bytes(img2_data);
    let descriptors1 = find_descriptors(&img1);
    let descriptors2 = find_descriptors(&img2);
    let matches = find_matches(descriptors1, descriptors2);
    let mut out :Vec<DescribedKeypoint> = Vec::new();
    for(desc1, desc2) in matches {
        let dk1 = desc1.into();
        let dk2 = desc2.into();
        out.push(dk1);
        out.push(dk2);
    }
    let count = out.len();
    let ptr: *const DescribedKeypoint = out.as_ptr();
    mem::forget(out);
    KeyVector {
        count,
        keys : ptr,
    }
}
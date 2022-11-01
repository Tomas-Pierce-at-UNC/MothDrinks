use pyo3::prelude::*;

use numpy::ndarray::array;
use numpy::array::PyArray2;
use numpy::ToPyArray;
use numpy::PyArray;
use pyo3::ToPyObject;

/// Formats the sum of two numbers as string.
#[pyfunction]
fn sum_as_string(a: usize, b: usize) -> PyResult<String> {
    Ok((a + b).to_string())
}

//                                          octave, scale, row, col, sigma, x, y, omega
#[pyfunction]
fn find_keypoints(image :&PyArray2<u8>) -> Vec<(u32,u32,u32,u32,f32,f32,f32,f32)> {
    unimplemented!()
}


/// A Python module implemented in Rust.
#[pymodule]
fn sift1(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(sum_as_string, m)?)?;
    m.add_function(wrap_pyfunction!(find_keypoints, m)?)?;
    Ok(())
}

mod histogram;

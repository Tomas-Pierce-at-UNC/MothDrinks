#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        let result = 2 + 2;
        assert_eq!(result, 4);
    }
}

mod histogram;

mod header;

use header::VideoHeaders;
use histogram::Histogram;

use std::fs;
use std::os::unix::io::FromRawFd;
use std::ptr;

use std::io::{self, Read, Seek, SeekFrom};

use std::mem;

fn read_frame(source: &mut fs::File, dest: &mut [u8], origin_offset: u64) -> io::Result<()> {
    let pos = SeekFrom::Start(origin_offset);
    source.seek(pos)?;
    let mut bytes: [u8; 4] = [0; 4];
    source.read_exact(&mut bytes)?;
    let forward: i64 = u32::from_le_bytes(bytes).into();
    source.seek(SeekFrom::Current(forward - 4))?;
    source.read_exact(dest)?;
    Ok(())
}

#[no_mangle]
pub unsafe extern "C" fn restricted_video_median(
    fd: i32,
    start_index: usize,
    img_count: usize,
) -> *const u8 {
    let mut file = fs::File::from_raw_fd(fd);
    let headers = match VideoHeaders::new(&mut file) {
        Some(h) => h,
        None => {
            return ptr::null();
        }
    };
    if start_index >= headers.image_count() || (start_index + img_count) >= headers.image_count() {
        return ptr::null();
    }
    let mut histograms = Vec::new();
    for _i in 0..headers.image_size() {
        let hist = Histogram::new();
        histograms.push(hist);
    }
    let offsets = match headers.get_image_offsets(&mut file) {
        Some(offs) => offs,
        None => {
            return ptr::null();
        }
    };
    let mut arena: Vec<u8> = vec![0; headers.image_size()];

    for index in start_index..(start_index + img_count) {
        let offset = offsets[index];
        match read_frame(&mut file, arena.as_mut_slice(), offset) {
            Ok(_a) => (),
            Err(_e) => {
                return ptr::null();
            }
        };
        for i in 0..headers.image_size() {
            let value = arena[i];
            histograms[i].update(value);
        }
    }

    let mut medians = Vec::with_capacity(histograms.len());
    for histogram in histograms {
        let median = histogram.median();
        medians.push(median);
    }
    medians.shrink_to_fit();
    let ptr = medians.as_ptr();
    mem::forget(medians);
    mem::forget(file);
    ptr
}

#[no_mangle]
pub unsafe extern "C" fn video_median(fd: i32) -> *const u8 {
    let mut file = fs::File::from_raw_fd(fd);
    let headers = match VideoHeaders::new(&mut file) {
        Some(h) => h,
        None => {
            return ptr::null();
        }
    };
    let mut histograms = Vec::new();
    for _i in 0..headers.image_size() {
        let hist = Histogram::new();
        histograms.push(hist);
    }
    let offsets = match headers.get_image_offsets(&mut file) {
        Some(offs) => offs,
        None => {
            return ptr::null();
        }
    };
    let mut arena: Vec<u8> = vec![0; headers.image_size()];
    for offset in offsets.into_iter() {
        /*
        let pos = SeekFrom::Start(*offset);
        match file.seek(pos) {
            Ok(_a) => (),
            Err(_e) => {return ptr::null();}
        };
        let mut bytes :[u8;4] = [0,0,0,0];
        match file.read_exact(&mut bytes) {
            Ok(_) => (),
            Err(_e) => {return ptr::null();},
        };
        let forward :i64 = u32::from_le_bytes(bytes).into();
        match file.seek(SeekFrom::Current(forward - 4)) {
            Ok(_) => (),
            Err(_e) => {return ptr::null();},
        };
        match file.read_exact(arena.as_mut_slice()) {
            Ok(_) => (),
            Err(_e) => {return ptr::null();},
        };
        */
        match read_frame(&mut file, arena.as_mut_slice(), *offset) {
            Ok(_a) => (),
            Err(_e) => {
                return ptr::null();
            }
        };

        for i in 0..headers.image_size() {
            let value = arena[i];
            histograms[i].update(value);
        }
    }

    let mut medians: Vec<u8> = Vec::new();
    for histogram in histograms {
        let median = histogram.median();
        medians.push(median);
    }
    medians.shrink_to_fit();
    let ptr = medians.as_ptr();
    mem::forget(medians);
    mem::forget(file);
    ptr
}

#[no_mangle]
pub unsafe extern "C" fn image_size(fd: i32) -> i32 {
    let mut file = fs::File::from_raw_fd(fd);
    let headers = match VideoHeaders::new(&mut file) {
        Some(h) => h,
        None => {
            return -1;
        }
    };
    let out = headers.image_size();
    mem::forget(file);
    out as i32
}

#[no_mangle]
pub unsafe extern "C" fn image_width(fd: i32) -> i32 {
    let mut file = fs::File::from_raw_fd(fd);
    let headers = match VideoHeaders::new(&mut file) {
        Some(h) => h,
        None => {
            return -1;
        }
    };

    let out = headers.image_width();
    mem::forget(file);
    out
}

#[no_mangle]
pub unsafe extern "C" fn image_height(fd: i32) -> i32 {
    let mut file = fs::File::from_raw_fd(fd);
    let headers = match VideoHeaders::new(&mut file) {
        Some(h) => h,
        None => {
            return -1;
        }
    };

    let out = headers.image_height();
    mem::forget(file);
    out
}

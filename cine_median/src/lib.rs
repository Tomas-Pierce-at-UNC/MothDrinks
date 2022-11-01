#[cfg(test)]
mod tests {

    use super::*;
    use std::os::unix::io::IntoRawFd;

    #[test]
    fn it_works() {
        let result = 2 + 2;
        assert_eq!(result, 4);
    }

    #[test]
    fn counts_images() {
        let mut file = fs::File::open("/home/tomas/Projects/DrinkMoth2/data2/moth22_2022-01-26.cine").unwrap();
        let headers = VideoHeaders::new(&mut file).unwrap();
        assert_eq!(headers.image_count(), 2000);
    }

    #[test]
    fn counts_image2() {
        let mut file = fs::File::open("/home/tomas/Projects/DrinkMoth2/data2/moth22_2022-01-26.cine").unwrap();
        let fd = file.into_raw_fd();
        let count = unsafe {
            image_count(fd)
        };
        let _f = unsafe {fs::File::from_raw_fd(fd) };
        assert_eq!(count, 2000);
    }

    #[test]
    fn last_image_works_bad() {
        let mut file = fs::File::open("/home/tomas/Projects/DrinkMoth2/bad_videos/moth22_2022-02-09_okay.cine").unwrap();
        let fd = file.into_raw_fd();
        let count = unsafe {
            image_count(fd)
        };
        let mut _f1 = unsafe {fs::File::from_raw_fd(fd)};
        let heads = VideoHeaders::new(&mut _f1).unwrap();
        let offsets = heads.get_image_offsets(&mut _f1).unwrap();
        let last_offset = offsets[(count - 1) as usize];
        let last_img = unsafe {
            read_frame_interop(fd, last_offset)
        };
        assert_ne!(last_img, ptr::null());
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
pub unsafe extern "C" fn read_frame_interop(fd: i32, offset: u64) -> *const u8 {
    let mut file = fs::File::from_raw_fd(fd);
    let headers = match VideoHeaders::new(&mut file) {
        Some(h) => h,
        None => {return ptr::null();}
    };
    let image_size = headers.image_size();
    let mut arena :Vec<u8> = vec![0u8;image_size];
    match read_frame(&mut file, &mut arena, offset) {
        Ok(()) => {
            arena.shrink_to_fit();
            let ptr = arena.as_ptr();
            mem::forget(file);
            mem::forget(arena);
            ptr
        },
        Err(_) => {
            mem::forget(file);
            ptr::null()
        }
    }
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
    if start_index >= headers.image_count() as usize || (start_index + img_count) >= headers.image_count() as usize {
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

#[no_mangle]
pub unsafe extern "C" fn image_count(fd :i32) -> u32 {
    let mut file = fs::File::from_raw_fd(fd);
    let chead = match VideoHeaders::new(&mut file) {
        Some(h) => h,
        None => {
            return 0;
        }
    };

    let out = chead.image_count() as u32;
    mem::forget(file);
    out
}
use std::fs;
use std::io::{Read, Seek, SeekFrom};
use std::mem;

#[derive(Copy, Clone, Debug)]
#[repr(C)]
pub struct CineHead {
    _kind: u16,
    _headersize: u16,
    _compression: u16,
    _version: u16,
    _first_movie_image: i32,
    _total_image_count: u32,
    _first_image_no: u32,
    _image_count: u32,
    _offset_image_header: u32,
    _offset_setup: u32,
    _offset_image_offsets: u32,
    _fractions: u32,
    _seconds: u32,
}

#[derive(Copy, Clone, Debug)]
#[repr(C)]
pub struct BitmapHead {
    _bi_size: u32,
    _bi_width: i32,
    _bi_height: i32,
    _bi_planes: u16,
    _bi_bit_count: u16,
    _bi_compression: u32,
    _bi_size_image: u32,
    _bi_x_pels_per_meter: i32,
    _bi_y_pels_per_meter: i32,
    _bi_clr_used: u32,
    _bi_clr_important: u32,
}

impl CineHead {
    const SIZE: usize = mem::size_of::<CineHead>();
}

impl BitmapHead {
    const SIZE: usize = mem::size_of::<BitmapHead>();
}

pub trait Takeable: Sized {
    const OFFSET: u64;
    fn take(file: &mut fs::File) -> Option<Self>;
}

impl Takeable for CineHead {
    const OFFSET: u64 = 0;
    fn take(file: &mut fs::File) -> Option<Self> {
        match file.seek(SeekFrom::Start(Self::OFFSET)).ok() {
            Some(it) => it,
            None => return None,
        };
        let mut data: [u8; Self::SIZE] = [0u8; Self::SIZE];
        match file.read_exact(&mut data).ok() {
            Some(it) => it,
            None => return None,
        };
        let me: CineHead = unsafe { *(data.as_ptr() as *const CineHead) };
        Some(me)
    }
}

impl Takeable for BitmapHead {
    const OFFSET: u64 = 0x2C;
    fn take(file: &mut fs::File) -> Option<Self> {
        match file.seek(SeekFrom::Start(Self::OFFSET)).ok() {
            Some(it) => it,
            None => return None,
        };
        let mut data: [u8; Self::SIZE] = [0u8; Self::SIZE];
        match file.read_exact(&mut data).ok() {
            Some(it) => it,
            None => return None,
        };
        let me: BitmapHead = unsafe { *(data.as_ptr() as *const BitmapHead) };
        Some(me)
    }
}

pub struct VideoHeaders {
    cine: CineHead,
    bitmap: BitmapHead,
}

impl VideoHeaders {
    pub fn new(file: &mut fs::File) -> Option<Self> {
        let cine = match CineHead::take(file) {
            Some(it) => it,
            None => return None,
        };
        let bitmap = match BitmapHead::take(file) {
            Some(it) => it,
            None => return None,
        };
        let heads = VideoHeaders { cine, bitmap };
        Some(heads)
    }

    pub fn image_size(&self) -> usize {
        self.bitmap._bi_size_image as usize
    }

    pub fn image_width(&self) -> i32 {
        self.bitmap._bi_width
    }

    pub fn image_height(&self) -> i32 {
        self.bitmap._bi_height
    }

    pub fn image_count(&self) -> usize {
        self.cine._image_count as usize
    }

    fn offset_to_image_offsets(&self) -> usize {
        self.cine._offset_image_offsets as usize
    }

    pub fn get_image_offsets(&self, file: &mut fs::File) -> Option<Box<[u64]>> {
        match file
            .seek(SeekFrom::Start(self.offset_to_image_offsets() as u64))
            .ok()
        {
            Some(it) => it,
            None => return None,
        };
        let mut arena = vec![0u8; self.image_count() * 8];
        match file.read_exact(arena.as_mut_slice()) {
            Ok(_blah) => (),
            Err(_e) => {
                return None;
            }
        };
        let mut offsets: Vec<u64> = Vec::new();
        for ch in arena.chunks(8) {
            let array: [u8; 8] = match ch.try_into() {
                Ok(arr) => arr,
                Err(_e) => {
                    return None;
                }
            };
            let offset = u64::from_le_bytes(array);
            offsets.push(offset);
        }
        let boxed = offsets.into_boxed_slice();
        Some(boxed)
    }
}

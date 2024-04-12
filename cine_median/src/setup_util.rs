
use std::io::{Read, Seek, SeekFrom};
use std::fs;

const FRAMERATE_OFFSET :u64 = 0x0054;

pub fn get_framerate(file :&mut fs::File) -> Option<u16> {
	file.seek(SeekFrom::Start(FRAMERATE_OFFSET)).ok()?;
	let mut bytes :[u8;2] = [0u8, 0u8];
	file.read_exact(&mut bytes).ok()?;
	let out :u16 = u16::from_le_bytes(bytes);
	Some(out)
}
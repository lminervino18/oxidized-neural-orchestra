// use rand::Rng;
// use std::ops::Range;
// use tokio::io::{AsyncRead, AsyncReadExt, BufReader};
//
// pub struct StreamSrc<R> {
//     reader: BufReader<R>,
//     buf: Vec<u8>,
//     ptr: usize,
// }
//
// impl<R: AsyncRead + Unpin> StreamSrc<R> {
//     pub fn new(src: R) -> Self {
//         todo!()
//     }
//
//     pub fn len(&self) -> usize {
//         todo!()
//     }
//
//     pub fn shuffle<Rn: Rng>(&self, rows: usize, row_size: usize, rng: &mut Rn) {
//         todo!()
//     }
//
//     pub fn raw_batch(&mut self, range: Range<usize>) -> &[f32] {
//         todo!()
//     }
// }
//
// #[cfg(test)]
// mod tests {
//     use super::*;
//     use std::io::Cursor;
//
//     #[test]
//     fn test_stream_src_raw_batch() {
//         let size = 127 * size_of::<f32>() as u64;
//         let raw_data: Vec<u8> = (0..size).map(|_| rand::rng().random()).collect();
//         let mut src = StreamSrc::new(Cursor::new(raw_data.clone()));
//
//         let (from, to) = (35, 84);
//
//         let expected: &[f32] =
//             bytemuck::cast_slice(&raw_data[from * size_of::<f32>()..to * size_of::<f32>()]);
//
//         let raw_batch = src.raw_batch(from..to);
//
//         assert_eq!(raw_batch, expected);
//     }
// }

use std::fs::File;
use std::io::Write;
use std::path::Path;

fn main() {
    let path = Path::new("dataset");
    let mut file = File::create(&path).unwrap();
    let nums: Vec<f32> = vec![1.0, 1.0, 2.0, 1.0, 2.0, 3.0, 4.0, 5.0, 9.0, 3.0, 6.0, 9.0];
    let mut bytes = vec![];
    nums.into_iter()
        .for_each(|n| bytes.extend_from_slice(&n.to_be_bytes()));
    let _ = file.write(&bytes).unwrap();
}

use chela::*;
use image::{GrayImage, Luma};
use std::path::Path;

pub fn load_png<P: AsRef<Path>>(path: P) -> NdArray<'static, i32> {
    let img = image::open(path).expect("Failed to open image");
    let gray = img.to_luma8();

    let (width, height) = gray.dimensions();

    let buffer: Vec<i32> = gray.pixels().map(|Luma([v])| *v as i32).collect();

    let data = NdArray::from(buffer);
    data.reshape([width as usize, height as usize])
}

pub fn load_data<P: AsRef<Path>>(path: P) -> NdArray<'static, f64> {
    let img = load_png(path).astype::<f64>();
    let img = (img / (255.0 / 2.0)) - 1.0;
    
    let size = img.size();
    img.reshape([1, size])
}

pub fn save_png(data: Vec<u8>, width: u32, height: u32, path: impl AsRef<Path>) {
    assert_eq!(data.len(), (width * height) as usize, "data size mismatch");
    
    let img: GrayImage = GrayImage::from_raw(width, height, data)
        .expect("Failed to create image buffer");
    
    img.save(path).expect("Failed to save PNG");
}

extern crate opencv;

use opencv::{
    core::Mat,
    core::Size,
    imgcodecs,
    types::VectorOfuchar,
    types::VectorOfint,
    prelude::Vector
};

pub type MatResult = Result<Mat, opencv::Error>;

pub fn resize(frame: &Mat, size: Size) -> MatResult {
    let mut target = Mat :: zeros(1, 1, frame.typ()?)?.to_mat()?;
    opencv::imgproc::resize(
        frame, &mut target, 
        size, 
        0f64, 0f64, 
        opencv::imgproc::INTER_NEAREST);
    Ok(target)
}

pub fn flip(frame: Mat) -> MatResult {
    let mut target = zeros(&frame)?;
    opencv::core::flip(&frame, &mut target, 0);
    Ok(target)
}

pub fn zeros(frame: &Mat) -> MatResult {    
    Mat :: zeros(frame.rows()?, frame.cols()?, frame.typ()?)?.to_mat()
}

pub fn decode_frame(buf: Vec<u8>) -> Result<Mat, opencv::Error> {
    // https://docs.rs/opencv/0.23.0/opencv/imgcodecs/fn.imdecode.html
    // https://vovkos.github.io/doxyrest-showcase/opencv/sphinxdoc/enum_cv_ImreadModes.html
    // https://docs.rs/crate/opencv/0.23.0/source/examples/warp_perspective_demo.rs

    //let original_image: Mat = imgcodecs::imread("image.png", imgcodecs::IMREAD_COLOR)?;
    let original_image: Mat = imgcodecs::imdecode(&VectorOfuchar :: from_iter(buf), imgcodecs::IMREAD_COLOR)?;
        
    let width = original_image.cols()?;
    let height = original_image.rows()?;

    //println!("Decoded frame, dimensions {}x{}", width, height);

    Ok(original_image)
}

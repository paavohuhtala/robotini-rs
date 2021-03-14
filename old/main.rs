extern crate opencv;

use opencv::{
    core::Mat,
    core::normalize,
    core::ToInputOutputArray,
    core::ToInputArray,
    imgcodecs,
    types::VectorOfint,
    prelude::Vector,
    types::VectorOfMat
};
use std::time::{SystemTime, UNIX_EPOCH};

extern crate serde;
extern crate serde_json;


#[cfg(not(any(feature = "raspi", feature="simulator")))]
mod local;
#[cfg(not(any(feature = "raspi", feature="simulator")))]
use local :: { connect, read_frame, send_command } ;


#[cfg(feature = "simulator")]
mod simulator;
#[cfg(feature = "simulator")]
use simulator :: { connect, read_frame, send_command } ;


#[cfg(feature = "raspi")]
mod raspi;
#[cfg(feature = "raspi")]
use raspi :: { connect, read_frame, send_command } ;

pub mod command;
use command :: { Command };


#[cfg(feature = "record")]
fn record_frame(orig_frame: &Mat, i: i32) {
    let image_name = format!("captures/frame{:04}.png", i);
    imgcodecs::imwrite(&image_name, &orig_frame, &VectorOfint::new());
}

#[cfg(not(feature = "record"))]
fn record_frame(orig_frame: &Mat, i: i32) {
    ()
}

pub mod image_processing;
use image_processing :: { flip, MatResult, decode_frame, resize };
use std::alloc::System;
use opencv::imgproc::{cvt_color, COLOR_BayerRG2GRAY, COLOR_BGR2GRAY, threshold, THRESH_BINARY, erode, get_structuring_element, MORPH_RECT, morphology_default_border_value};
use opencv::core::{ToOutputArray, NORM_MINMAX, no_array, _InputArrayTrait, BORDER_CONSTANT, Size, Point_, bitwise_and, norm, split, Rect, Rect_, mean, count_non_zero, abs};
use std::ptr::null;
use std::fs::File;
use std::io::Write;

fn main() -> Result<(), Box<std::error::Error>> {
    let mut conn = connect()?;
    let mut wheels_turn = 0.0f32;
    let mut speed = 0.0f32;
    let mut i = 0i32;

    let mut log_file = File::create("log.csv")?;
    log_file.write("blue_ratio;red_ratio;green_ratio;wheel_turn;speed;time;\n".as_bytes());

    loop {
        let orig_frame = read_frame(&mut conn)?;
        record_frame(&orig_frame, i);

        i += 1;

        let frames = process_frame(orig_frame);
        //get_action(&frames); (r, rgreen, rblue, rred)
        //println!("blue frame rows: {}", frames.get(0).unwrap().1.rows().ok().unwrap());
        let blue_roi = Mat::roi(&frames.get(0).unwrap().1, Rect_ {x: 32, y: 20, width: 64, height: 60}).ok().unwrap();
        save_frame_to_file("blue-roi", &blue_roi);
        let blue_count = count_non_zero(&blue_roi).unwrap() as f32;

        // calc the green frame and if we should move left
        //println!("green frame cols: {}", frames.get(1).unwrap().1.cols().ok().unwrap());
        let green_roi = Mat::roi(&frames.get(1).unwrap().2, Rect_ {x: 64, y: 20, width: 64, height: 60}).ok().unwrap();
        save_frame_to_file("green-roi", &green_roi);
        let green_count = count_non_zero(&green_roi).unwrap() as f32;

        // calc the red frame and if we should move right
        //println!("red frame cols: {}", frames.get(2).unwrap().1.cols().ok().unwrap());
        let red_roi = Mat::roi(&frames.get(2).unwrap().3, Rect_ {x: 0, y: 20, width: 64, height: 60}).ok().unwrap();
        save_frame_to_file("red-roi", &red_roi);
        let red_count = count_non_zero(&red_roi).unwrap() as f32;
        //println!("red count: {}", &count / (64.0*60.0));

        let red_ratio = red_count / (64.0*60.0);
        let green_ratio = green_count / (64.0*60.0);
        let blue_ratio = blue_count / (64.0*60.0);
        //println!("green count: {}", &green_count / (64.0*60.0));

        let diff = red_ratio - green_ratio;
        if blue_ratio < 0.6 {
            wheels_turn = (wheels_turn - diff * 2.0f32).max(-0.9f32).min(0.9f32);
        }
        let max_speed = 0.15;
        let min_speed = 0.06;
        speed = (0.01 / wheels_turn.abs().max(0.01)).min(max_speed).max(min_speed);

        let timestamp = SystemTime::now().duration_since(UNIX_EPOCH).ok().unwrap();
        log_file.write(format!("{};{};{};{};{};{};\n", &blue_ratio, &red_ratio, &green_ratio, &wheels_turn, &speed, timestamp.as_millis()).as_bytes());
/*
        if red_ratio > 0.3 {
            wheels_turn = (wheels_turn + 0.2f32).min(1.0f32);
        } else if green_ratio > 0.3 {
            wheels_turn = (wheels_turn - 0.2f32).max(-1.0f32);
        }
*/

        send_command(&Command::Move(true), &mut conn);
        send_command(&Command::Turn(wheels_turn), &mut conn);
        send_command(&Command::Forward(speed), &mut conn);

        wheels_turn *= 0.4;

        //::std::process::exit(1);
    }
}

fn save_frame_to_file(name: &str, frame: &Mat) {
    let timestamp = SystemTime::now().duration_since(UNIX_EPOCH).ok().unwrap();
    let image_name = format!("captures/{}-{}.png", timestamp.as_millis(), name);
    #[cfg(feature = "dump")]
    imgcodecs::imwrite(&image_name, frame, &VectorOfint :: new());
}

fn process_frame(frame: Mat) -> Vec<(Mat, Mat, Mat, Mat)> {
    save_frame_to_file("original", &frame);

    // calculate the blacks
    let mut gray_scaled = frame.clone().ok().unwrap();
    cvt_color(
        &frame,
        &mut gray_scaled,
        COLOR_BGR2GRAY,
        0
    );
    save_frame_to_file("graycolor", &gray_scaled);

    let mut th = frame.clone().ok().unwrap();
    threshold(&mut gray_scaled, &mut th, 30.0, 255.0, THRESH_BINARY);

    let mut blacks = frame.clone().ok().unwrap();
    erode(&th,
          &mut blacks,
          &get_structuring_element(MORPH_RECT, Size{width: 2, height: 2}, Point_{x: -1, y: -1}).ok().unwrap(),
          Point_{x: -1, y: -1},
          1,
          BORDER_CONSTANT,
          morphology_default_border_value().ok().unwrap());

    let mask = Mat::default();
    let mut normalized = frame.clone().ok().unwrap();
    normalize(&frame, &mut normalized, 0.0, 255.0, NORM_MINMAX, -1, &mask.ok().unwrap());

    let mut preprosessed_image = frame.clone().ok().unwrap();
    bitwise_and(&normalized, &normalized, &mut preprosessed_image, &blacks).ok();

    let mut split_frame_red = frame.clone().ok().unwrap();
    let mut split_frame_green = frame.clone().ok().unwrap();
    let mut split_frame_blue = frame.clone().ok().unwrap();
    let mut split_frame = VectorOfMat::new();
    split_frame.push(split_frame_blue);
    split_frame.push(split_frame_green);
    split_frame.push(split_frame_red);
    split(&preprosessed_image, &mut split_frame);

    // erode the blue green and red
    let mut split_frame_processed : Vec<(Mat, Mat, Mat, Mat)> = split_frame
        .iter()
        .map(| c | {
            let mut r = c.clone().ok().unwrap();
            erode(&c,
                  &mut r,
                  &get_structuring_element(MORPH_RECT, Size{width: 2, height: 2}, Point_{x: -1, y: -1}).ok().unwrap(),
                  Point_{x: -1, y: -1},
                  1,
                  BORDER_CONSTANT,
                  morphology_default_border_value().ok().unwrap());

            // 100 is good for red, blue and green is good with 120
            let mut rred = c.clone().ok().unwrap();
            threshold(&r, &mut rred, 200.0, 255.0, THRESH_BINARY);

            let mut rgreen = c.clone().ok().unwrap();
            threshold(&r, &mut rgreen, 200.0, 255.0, THRESH_BINARY);

            let mut rblue = c.clone().ok().unwrap();
            threshold(&r, &mut rblue, 225.0, 255.0, THRESH_BINARY);

            (r, rblue, rgreen, rred)
        }).collect();

    save_frame_to_file("blue-0", &split_frame_processed.get(0).unwrap().0);
    save_frame_to_file("blue-1", &split_frame_processed.get(0).unwrap().1);
    save_frame_to_file("green-0", &split_frame_processed.get(1).unwrap().0);
    save_frame_to_file("green-1", &split_frame_processed.get(1).unwrap().2);
    save_frame_to_file("red-0", &split_frame_processed.get(2).unwrap().0);
    save_frame_to_file("red-1", &split_frame_processed.get(2).unwrap().3);

    split_frame_processed
}

use std::collections::VecDeque;

use opencv::{
    core::{
        bitwise_and, count_non_zero, normalize, split, Point_, Rect_, Scalar_, Size,
        BORDER_CONSTANT, NORM_MINMAX,
    },
    highgui, imgcodecs,
    imgproc::{
        cvt_color, erode, get_structuring_element, morphology_default_border_value, threshold,
        COLOR_BGR2GRAY, LINE_8, MORPH_RECT, THRESH_BINARY,
    },
    prelude::*,
    types::{VectorOfMat, VectorOfu8},
};

mod connection;
use connection::{Command, Connection, LoginMessage};

const DEBUG_SAVE_IMAGES: bool = false;
const DEBUG_GUI: bool = false;

fn save_frame(mat: &Mat, i: usize) -> anyhow::Result<()> {
    let image_name = format!("captures/frame{:04}.png", i);
    save_frame_to_file(&image_name, mat)
}

fn save_frame_to_file(name: &str, mat: &Mat) -> anyhow::Result<()> {
    if !DEBUG_SAVE_IMAGES {
        return Ok(());
    }

    imgcodecs::imwrite(&name, &mat, &opencv::core::Vector::<i32>::new())?;
    Ok(())
}

fn run() -> anyhow::Result<()> {
    std::fs::create_dir_all("captures/debug")?;

    let team_id = std::env::var("teamid").unwrap_or(String::from("rust"));
    let address = std::env::var("SIMULATOR").unwrap_or(String::from("127.0.0.1:11000"));

    let mut connection = Connection::connect(
        &address,
        &LoginMessage {
            name: "Team Rust",
            color: "#ff9514",
            team_id: &team_id,
        },
    )?;

    if DEBUG_GUI {
        let window = "robotini";
        highgui::named_window(window, 1)?;
    }

    let mut frame_i = 0;
    let mut car_state = CarState {
        speed: 0.0,
        wheels_turn: 0.0,
        previous_horizons: VecDeque::new(),
    };

    loop {
        let image = connection.read_next_image()?;

        let frame =
            opencv::imgcodecs::imdecode(&VectorOfu8::from(image), opencv::imgcodecs::IMREAD_COLOR)?;

        save_frame(&frame, frame_i)?;
        frame_update(&frame, &mut car_state, &mut &mut connection)?;

        connection.send(&Command::Forward { value: 0.15 })?;

        if DEBUG_GUI {
            let key = highgui::wait_key(10)?;
            if key > 0 && key != 255 {
                break;
            }
        }

        frame_i += 1;
    }
    Ok(())
}

struct CarState {
    wheels_turn: f32,
    speed: f32,
    previous_horizons: VecDeque<i32>,
}

fn frame_update(
    frame: &Mat,
    state: &mut CarState,
    connection: &mut Connection,
) -> anyhow::Result<()> {
    let wheels_turn = &mut state.wheels_turn;
    let speed = &mut state.speed;

    let frames = process_frame(frame)?;

    let (horizon_i, _horizon_blackness) = {
        let filtered = &frames[2].3;
        let cols = filtered.cols();

        (0..filtered.rows() / 2)
            .rev()
            .map(|y| {
                let row = filtered.at_row::<u8>(y).unwrap();
                let black_pixel_count = row.iter().filter(|px| **px == 0).count();
                let fullness = black_pixel_count as f32 / cols as f32;
                (y, fullness)
            })
            .max_by(|(_, fullness_a), (_, fullness_b)| {
                PartialOrd::partial_cmp(fullness_a, fullness_b).unwrap()
            })
    }
    .unwrap();

    state.previous_horizons.push_front(horizon_i);
    state.previous_horizons.truncate(60);
    let horizon_interpolated = (state.previous_horizons.iter().sum::<i32>() as f32
        / state.previous_horizons.len() as f32) as i32;

    let width = frames[0].1.cols();
    let height = frames[0].1.rows();

    let roi_rect = Rect_ {
        x: 0,
        y: horizon_interpolated,
        width,
        height: height - horizon_interpolated,
    };

    let total_pixels = roi_rect.width * roi_rect.height;

    let blue_roi = Mat::roi(&frames.get(0).unwrap().1, roi_rect).ok().unwrap();
    save_frame_to_file("captures/debug/blue-roi.png", &blue_roi)?;
    let blue_count = count_non_zero(&blue_roi).unwrap() as f32;

    // calc the green frame and if we should move left
    let green_roi = Mat::roi(&frames.get(1).unwrap().2, roi_rect).ok().unwrap();
    save_frame_to_file("captures/debug/green-roi.png", &green_roi)?;
    let green_count = count_non_zero(&green_roi).unwrap() as f32;

    // calc the red frame and if we should move right
    let red_roi = Mat::roi(&frames.get(2).unwrap().3, roi_rect).ok().unwrap();
    save_frame_to_file("captures/debug/red-roi.png", &red_roi)?;
    let red_count = count_non_zero(&red_roi).unwrap() as f32;

    let red_ratio = red_count / total_pixels as f32;
    let green_ratio = green_count / total_pixels as f32;
    let blue_ratio = blue_count / total_pixels as f32;

    let diff = red_ratio - green_ratio;
    if blue_ratio < 0.6 {
        *wheels_turn = (*wheels_turn - diff * 1.8f32).max(-0.9f32).min(0.9f32);
    }
    let max_speed = 0.03;
    let min_speed = 0.002;
    *speed = (0.001 / wheels_turn.abs().max(0.01))
        .min(max_speed)
        .max(min_speed);

    connection.send(&Command::Forward { value: *speed })?;
    connection.send(&Command::Turn {
        value: *wheels_turn,
    })?;

    *wheels_turn *= 0.3;

    let mut viz_frame = frame.clone();

    if DEBUG_GUI {
        opencv::imgproc::line(
            &mut viz_frame,
            Point_ {
                x: 0,
                y: horizon_interpolated,
            },
            Point_ {
                x: 200,
                y: horizon_interpolated,
            },
            Scalar_([0.0, 0.0, 1.0, 0.0]),
            1,
            LINE_8,
            0,
        )?;

        highgui::imshow("robotini", &viz_frame)?;
    }

    Ok(())
}

fn process_frame(frame: &Mat) -> anyhow::Result<Vec<(Mat, Mat, Mat, Mat)>> {
    save_frame_to_file("captures/debug/original.png", frame)?;

    // calculate the blacks
    let mut gray_scaled = frame.clone();
    cvt_color(&frame, &mut gray_scaled, COLOR_BGR2GRAY, 0)?;
    save_frame_to_file("captures/debug/graycolor.png", &gray_scaled)?;

    let mut th = frame.clone();
    threshold(&mut gray_scaled, &mut th, 30.0, 255.0, THRESH_BINARY)?;

    let mut blacks = frame.clone();
    erode(
        &th,
        &mut blacks,
        &get_structuring_element(
            MORPH_RECT,
            Size {
                width: 2,
                height: 2,
            },
            Point_ { x: -1, y: -1 },
        )
        .ok()
        .unwrap(),
        Point_ { x: -1, y: -1 },
        1,
        BORDER_CONSTANT,
        morphology_default_border_value().ok().unwrap(),
    )?;

    let mask = Mat::default();
    let mut normalized = frame.clone();
    normalize(
        &frame,
        &mut normalized,
        0.0,
        255.0,
        NORM_MINMAX,
        -1,
        &mask.ok().unwrap(),
    )?;

    let mut preprosessed_image = frame.clone();
    bitwise_and(&normalized, &normalized, &mut preprosessed_image, &blacks).ok();

    let split_frame_red = frame.clone();
    let split_frame_green = frame.clone();
    let split_frame_blue = frame.clone();
    let mut split_frame = VectorOfMat::new();
    split_frame.push(split_frame_blue);
    split_frame.push(split_frame_green);
    split_frame.push(split_frame_red);
    split(&preprosessed_image, &mut split_frame)?;

    // erode the blue green and red
    let split_frame_processed: Vec<(Mat, Mat, Mat, Mat)> = split_frame
        .iter()
        .map(|c| {
            let mut r = c.clone();
            erode(
                &c,
                &mut r,
                &get_structuring_element(
                    MORPH_RECT,
                    Size {
                        width: 2,
                        height: 2,
                    },
                    Point_ { x: -1, y: -1 },
                )
                .ok()
                .unwrap(),
                Point_ { x: -1, y: -1 },
                1,
                BORDER_CONSTANT,
                morphology_default_border_value().ok().unwrap(),
            )
            .unwrap();

            // 100 is good for red, blue and green is good with 120
            let mut rred = c.clone();
            threshold(&r, &mut rred, 150.0, 255.0, THRESH_BINARY).unwrap();

            let mut rgreen = c.clone();
            threshold(&r, &mut rgreen, 200.0, 255.0, THRESH_BINARY).unwrap();

            let mut rblue = c.clone();
            threshold(&r, &mut rblue, 200.0, 255.0, THRESH_BINARY).unwrap();

            (r, rblue, rgreen, rred)
        })
        .collect();

    save_frame_to_file(
        "captures/debug/blue-0.png",
        &split_frame_processed.get(0).unwrap().0,
    )?;
    save_frame_to_file(
        "captures/debug/blue-1.png",
        &split_frame_processed.get(0).unwrap().1,
    )?;
    save_frame_to_file(
        "captures/debug/green-0.png",
        &split_frame_processed.get(1).unwrap().0,
    )?;
    save_frame_to_file(
        "captures/debug/green-1.png",
        &split_frame_processed.get(1).unwrap().2,
    )?;
    save_frame_to_file(
        "captures/debug/red-0.png",
        &split_frame_processed.get(2).unwrap().0,
    )?;
    save_frame_to_file(
        "captures/debug/red-1.png",
        &split_frame_processed.get(2).unwrap().3,
    )?;

    Ok(split_frame_processed)
}

fn main() {
    run().unwrap()
}

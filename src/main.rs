use opencv::{
    core::{bitwise_and, normalize, split, Point_, Size, BORDER_CONSTANT, NORM_MINMAX},
    highgui, imgcodecs,
    imgproc::{
        cvt_color, erode, get_structuring_element, morphology_default_border_value, threshold,
        COLOR_BGR2GRAY, MORPH_RECT, THRESH_BINARY,
    },
    prelude::*,
    types::{VectorOfMat, VectorOfu8},
};

mod connection;
use connection::{Command, Connection, LoginMessage};

fn save_frame(mat: &Mat, i: usize) -> anyhow::Result<()> {
    let image_name = format!("captures/frame{:04}.png", i);
    save_frame_to_file(&image_name, mat)
}

fn save_frame_to_file(name: &str, mat: &Mat) -> anyhow::Result<()> {
    imgcodecs::imwrite(&name, &mat, &opencv::core::Vector::<i32>::new())?;
    Ok(())
}

fn run() -> anyhow::Result<()> {
    std::fs::create_dir_all("captures/debug")?;

    let mut connection = Connection::connect(
        "127.0.0.1:11000",
        &LoginMessage {
            name: "Team Rust",
            color: "#ff9514",
            team_id: "rust",
        },
    )?;

    let window = "video capture";
    highgui::named_window(window, 1)?;

    let mut frame_i = 0;

    loop {
        let image = connection.read_next_image()?;

        let mut frame =
            opencv::imgcodecs::imdecode(&VectorOfu8::from(image), opencv::imgcodecs::IMREAD_COLOR)?;

        save_frame(&frame, frame_i)?;
        process_frame(&frame)?;

        connection.send(&Command::Forward { value: 0.15 })?;

        if frame.size()?.width > 0 {
            highgui::imshow(window, &mut frame)?;
        }

        let key = highgui::wait_key(10)?;
        if key > 0 && key != 255 {
            break;
        }

        frame_i += 1;
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
            threshold(&r, &mut rred, 200.0, 255.0, THRESH_BINARY).unwrap();

            let mut rgreen = c.clone();
            threshold(&r, &mut rgreen, 200.0, 255.0, THRESH_BINARY).unwrap();

            let mut rblue = c.clone();
            threshold(&r, &mut rblue, 225.0, 255.0, THRESH_BINARY).unwrap();

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

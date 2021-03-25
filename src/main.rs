use opencv::{highgui, imgcodecs, prelude::*, types::VectorOfu8};

mod connection;
use connection::{Command, Connection, LoginMessage};

fn save_frame(mat: &Mat, i: usize) -> anyhow::Result<()> {
    let image_name = format!("captures/frame{:04}.png", i);
    imgcodecs::imwrite(&image_name, &mat, &opencv::core::Vector::<i32>::new())?;
    Ok(())
}

fn run() -> anyhow::Result<()> {
    std::fs::create_dir_all("captures")?;

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

fn main() {
    run().unwrap()
}

use opencv::{highgui, prelude::*, types::VectorOfu8};

mod connection;
use connection::{Command, Connection, LoginMessage};

fn run() -> anyhow::Result<()> {
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

    loop {
        let image = connection.read_next_image()?;

        let mut frame =
            opencv::imgcodecs::imdecode(&VectorOfu8::from(image), opencv::imgcodecs::IMREAD_COLOR)?;

        connection.send(&Command::Forward { value: 0.15 })?;

        if frame.size()?.width > 0 {
            highgui::imshow(window, &mut frame)?;
        }

        let key = highgui::wait_key(10)?;
        if key > 0 && key != 255 {
            break;
        }
    }
    Ok(())
}

fn main() {
    run().unwrap()
}

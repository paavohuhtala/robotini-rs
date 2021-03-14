use std::io:: { stdout, Stdout };
use std::fs;
use std::fs::File;
use std::io::prelude::*;
use command :: { Command, send as _send };
use std::os::unix::net::UnixStream;
use image_processing :: { decode_frame, flip };
use opencv::core::Mat;
type Stream = UnixStream;
type Camera = ();
type Connection = (Stream, Camera, i32);

pub fn send_command(command: &Command, conn: &mut Connection) -> Result<(), std::io::Error> {
    _send(command, &mut conn.0)
}

pub fn connect() -> Result<Connection, std::io::Error> {
    let mut stream = UnixStream::connect("/tmp/motor-server.socket")?;
    println!("Connected to motor server");
    Ok((stream, (), 1i32))
}

pub fn read_frame(camera: &mut Connection) -> Result<Mat, std::io::Error> {
    let filename = format!("test-images/{}.jpg", camera.2);
    //println!("frame: {}", filename);
    camera.2 = ((camera.2 + 1) % 290) + 1;
    let attr = fs::metadata(&filename)?;
    let mut f = File::open(&filename)?;
    let len = attr.len();
    let mut buffer:Vec<u8> = vec!(0u8; len as usize);
    f.read_exact(&mut buffer);
    Ok(decode_frame(buffer).unwrap())
}

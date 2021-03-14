extern crate rascam;
use self::rascam::*;

use std::fs::File;
use std::io::Write;
use std::{thread, time};
use std::io:: { stdout, Stdout };
use std::fs;
use std::convert::From;
use std::io::prelude::*;
use command :: { Command, send as _send };
use std::os::unix::net::UnixStream;
use image_processing :: { decode_frame, flip, resize };
use opencv::core::{ Mat, Size, CV_8UC3, Vec3 };

type Stream = UnixStream;
type Camera = SeriousCamera;
type Connection = (Stream, Stream);

const WIDTH: u32 = 128;
const HEIGHT: u32 = 80;

pub fn connect() -> Result<Connection, std::io::Error> {
    let mut camera = UnixStream::connect("/tmp/camera.sock")?;
    let mut stream = UnixStream::connect("/tmp/motor-server.socket")?;
    println!("Connected to motor server");

    Ok((stream, camera))
}

pub fn send_command(command: &Command, conn: &mut Connection) -> Result<(), std::io::Error> {
    _send(command, &mut conn.0)
}

pub fn read_frame(conn: &mut Connection) -> Result<Mat, std::io::Error> {
    let stream = &mut conn.1;
    let len = (WIDTH * HEIGHT * 3) as usize;
    let mut bytes: Vec<u8> = vec![0u8; len];
    stream.read_exact(&mut bytes);

    let mut buf = {
	    let mut data = Vec::<Vec3<u8>>::new();
	    for y in 0..HEIGHT {
			for x in 0..WIDTH {
				let idx = ((y * WIDTH + x) * 3) as usize;
				let v: [u8; 3] = [bytes[idx], bytes[idx+1], bytes[idx+2]];
				data.push(Vec3::from(v))
			}
	    }

		data
	};

    let decoded = Mat::from_slice(&mut buf).unwrap().reshape(3, HEIGHT as i32).unwrap();

    Ok(decoded)
}


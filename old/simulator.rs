extern crate byteorder;

use self::byteorder::{BigEndian, ReadBytesExt};
use std::io::prelude::*;
use std::net::TcpStream;
use command :: { Command, send as _send };
use image_processing :: { decode_frame };
use opencv::core::Mat;

type Stream = TcpStream;

pub fn send_command(command: &Command, stream: &mut Stream) -> Result<(), std::io::Error> {
    _send(command, stream)
}

pub fn connect() -> Result<Stream, std::io::Error> {
    let host = "localhost:11000";
    // https://doc.rust-lang.org/std/net/struct.TcpStream.html
    let mut stream = TcpStream::connect(host)?; // TODO error handling
    Ok(stream)
}

pub fn read_frame(stream: &mut Stream) -> Result<Mat, std::io::Error> {
    let len = stream.read_u16 ::<BigEndian>()?;
    // https://www.joshmcguigan.com/blog/array-initialization-rust/
    let mut buf: Vec<u8> = vec![0u8; len as usize];
    // https://doc.rust-lang.org/std/io/trait.Read.html#method.read_exact
    stream.read_exact(&mut buf);
    Ok(decode_frame(buf).unwrap())
}

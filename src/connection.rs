use std::{
    io::{Read, Write},
    net::{SocketAddr, TcpStream},
    time::Duration,
};

use serde::Serialize;

#[derive(Serialize)]
#[serde(tag = "action")]
#[serde(rename_all = "lowercase")]
pub enum Command {
    Forward { value: f32 },
    Turn { value: f32 },
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct LoginMessage<'a> {
    pub name: &'a str,
    pub color: &'a str,
    pub team_id: &'a str,
}

pub struct Connection {
    tcp_stream: TcpStream,
}

impl Connection {
    pub fn connect(address: &str, login_message: &LoginMessage) -> anyhow::Result<Connection> {
        let address: SocketAddr = address.parse().unwrap();
        let tcp_stream = TcpStream::connect_timeout(&address, Duration::from_secs(2));
        let mut tcp_stream = tcp_stream.expect("Expected Robotini server to answer");

        serde_json::to_writer(&mut tcp_stream, login_message)?;

        tcp_stream.write_all(&[b'\n'])?;

        Ok(Connection { tcp_stream })
    }

    pub fn send(&mut self, command: &Command) -> anyhow::Result<()> {
        serde_json::to_writer(&mut self.tcp_stream, command)?;
        self.tcp_stream.write_all(&[b'\n'])?;
        Ok(())
    }

    pub fn read_next_image(&mut self) -> anyhow::Result<Vec<u8>> {
        let mut length = [0u8; 2];
        self.tcp_stream.read_exact(&mut length)?;
        let length = u16::from_be_bytes(length) as usize;

        let mut buffer = vec![0u8; length];
        self.tcp_stream.read_exact(&mut buffer)?;

        Ok(buffer)
    }
}

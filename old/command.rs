extern crate serde;
extern crate serde_json;

use serde::{
    Deserialize, 
    Serialize
};

use serde_json::to_string as to_json_string;
use std::io::prelude::*;
use std::net::TcpStream;


#[derive(Serialize, Deserialize)]
struct MoveJsonCommand {
    r#move: bool,
}

#[derive(Serialize, Deserialize)]
struct ActionJsonCommand {
    action: String,
    value: f32
}

pub enum Command {
    Move(bool),
    Forward(f32),
    Turn(f32),
    Brake(f32)
}

pub fn send(command: &Command, stream: &mut impl Write) -> Result<(), std::io::Error> {
    let command_json: String = match command {
        Command::Move(can_move) => to_json_string(&MoveJsonCommand { r#move: *can_move }),
        Command::Forward(value) => to_json_string(&ActionJsonCommand { action: "forward".to_string(), value: *value }),
        Command::Turn(value) => to_json_string(&ActionJsonCommand { action: "turn".to_string(), value: *value }),
        Command::Brake(value) => to_json_string(&ActionJsonCommand { action: "brake".to_string(), value: *value })        
    }?;    
    stream.write(command_json.as_bytes());
    stream.write(b"\n");
    stream.flush();
    Ok(())
}

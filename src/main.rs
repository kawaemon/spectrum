#![feature(let_chains)]

use std::time::{Duration, Instant};

use sdl2::{
    audio::{AudioFormat, AudioSpecDesired},
    event::Event,
    keyboard::Keycode,
    mixer::{InitFlag, AUDIO_F32LSB},
};

use crate::render::Renderer;

use self::player::Player;

mod fft;
mod player;
mod render;
mod tone;

const WIDTH: u32 = 800;
const HEIGHT: u32 = 600;

const PLAYBACK_SKIP: usize = (40 * FREQUENCY) as usize;
const FREQUENCY: i32 = 48000;
const FORMAT: u16 = AUDIO_F32LSB;
const CHANNELS: i32 = 2;
const CHUNK_SIZE: i32 = 1024;

fn main() {
    let sdl = sdl2::init().unwrap();
    let video = sdl.video().unwrap();
    let audio = sdl.audio().unwrap();

    sdl2::mixer::open_audio(FREQUENCY, FORMAT, CHANNELS, CHUNK_SIZE).unwrap();
    let _mixer_context = sdl2::mixer::init(InitFlag::MP3).unwrap();

    let wav = hound::WavReader::open("./out.wav").unwrap();
    let wavspec = wav.spec();
    assert_eq!(wavspec.channels, 2);

    let (samples, _) = wav
        .into_samples::<f32>()
        .skip(PLAYBACK_SKIP)
        .try_fold((vec![], true), |(mut samples, is_left), v| {
            let v = v? as f64;
            if is_left {
                samples.push(v);
            } else {
                *samples.last_mut().unwrap() += v;
            }
            Result::<_, hound::Error>::Ok((samples, !is_left))
        })
        .unwrap();

    let window = video
        .window("spectrum", WIDTH, HEIGHT)
        .position_centered()
        .opengl()
        .build()
        .unwrap();

    let mut canvas = window.into_canvas().build().unwrap();

    let desired_spec = AudioSpecDesired {
        freq: Some(FREQUENCY),
        channels: Some(1),
        samples: None,
    };

    let played = samples.clone();
    let device = audio
        .open_playback(None, &desired_spec, |spec| {
            assert_eq!(spec.format, AudioFormat::F32LSB);
            assert_eq!(spec.channels, 1);
            Player::new(played)
        })
        .unwrap();

    let mut event_pump = sdl.event_pump().unwrap();
    device.resume();
    let began_at = Instant::now();

    std::thread::sleep(Duration::from_secs_f64(0.5));

    let mut renderer = Renderer::new();

    'main: loop {
        for event in event_pump.poll_iter() {
            match event {
                Event::Quit { .. }
                | Event::KeyDown {
                    keycode: Some(Keycode::Q | Keycode::Escape),
                    ..
                } => break 'main,
                _ => {}
            }
        }

        renderer.render(&mut canvas, wavspec.sample_rate, began_at, &samples);
        std::thread::sleep(Duration::from_secs_f64(1.0 / 60.0));
    }
}

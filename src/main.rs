use rustfft::{
    num_complex::{Complex, ComplexFloat},
    FftPlanner,
};
use sdl2::{
    event::Event,
    keyboard::Keycode,
    mixer::{InitFlag, Music, AUDIO_F32LSB},
    pixels::Color,
};

const WIDTH: u32 = 1280;
const HEIGHT: u32 = 420;

const FREQUENCY: i32 = 48000;
const FORMAT: u16 = AUDIO_F32LSB;
const CHANNELS: i32 = 2;
const CHUNK_SIZE: i32 = 1024;

// https://ryo-iijima.com/fftresult/
fn fft(signal: &[f64], sampling_freq: usize) -> Vec<(f64, f64)> {
    let n = signal.len();

    let fft_data = {
        let mut signal = signal
            .iter()
            .map(|&x| Complex { re: x, im: 0.0 })
            .collect::<Vec<_>>();
        FftPlanner::new()
            .plan_fft_forward(signal.len())
            .process(&mut signal);
        signal
    };

    let delta_f = sampling_freq as f64 / n as f64;

    // https://github.com/numpy/numpy/blob/v1.24.0/numpy/fft/helper.py#L172-L221
    let fft_freq = {
        let n = n as f64;
        let d = 1.0 / sampling_freq as f64;
        let val = 1.0 / (n * d);
        let cn = (n as usize / 2) + 1;
        let results = 0..cn;
        results.map(|x| (x as f64) * val).collect::<Vec<_>>()
    };

    let fft_psd_db = fft_data
        .into_iter()
        .map(|data| data.abs())
        .map(|as_| as_.powf(2.0))
        .map(|ps| ps / delta_f)
        .map(|psd| 10.0 * psd.log10())
        .collect::<Vec<_>>();

    fft_freq.into_iter().zip(fft_psd_db).collect()
}

#[test]
fn test_fft() {
    use std::f64::consts::PI;

    let signal = (0..100)
        .map(|tick| {
            let sec = tick as f64 / 10.0;
            f64::sin(2.0 * PI * sec)
        })
        .collect::<Vec<_>>();

    for (i, (f, v)) in fft(&signal, 10).into_iter().enumerate() {
        if i == 10 {
            assert!((f - 1.0).abs() < 0.0000001);
            assert!(v > 0.0);
        } else {
            assert!(v < 0.0);
        }
    }
}

fn main() {
    let sdl = sdl2::init().unwrap();
    let video = sdl.video().unwrap();
    let _audio = sdl.audio().unwrap();

    sdl2::mixer::open_audio(FREQUENCY, FORMAT, CHANNELS, CHUNK_SIZE).unwrap();
    let _mixer_context = sdl2::mixer::init(InitFlag::MP3).unwrap();

    let sdl_source = Music::from_file("./d.mp3").unwrap();

    let _wav = hound::WavReader::open("./d.wav")
        .unwrap()
        .into_samples::<f32>()
        .collect::<Vec<_>>();

    sdl_source.play(0).unwrap();

    let window = video
        .window("spectrum", WIDTH, HEIGHT)
        .position_centered()
        .opengl()
        .build()
        .unwrap();

    let mut canvas = window.into_canvas().build().unwrap();

    canvas.set_draw_color(Color::BLACK);
    canvas.clear();
    canvas.present();

    let mut event_pump = sdl.event_pump().unwrap();

    'render: loop {
        for event in event_pump.poll_iter() {
            match event {
                Event::Quit { .. }
                | Event::KeyDown {
                    keycode: Some(Keycode::Q | Keycode::Escape),
                    ..
                } => break 'render,
                _ => {}
            }
        }
    }
}

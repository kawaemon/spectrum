#![feature(let_chains)]

use std::{
    collections::VecDeque,
    time::{Duration, Instant},
};

use color::{color_space::LinearRgb, Deg, Hsv, Rgb, ToRgb};
use rustfft::{
    num_complex::{Complex, ComplexFloat},
    FftPlanner,
};
use sdl2::{
    audio::{AudioFormat, AudioSpecDesired},
    event::Event,
    keyboard::Keycode,
    mixer::{InitFlag, AUDIO_F32LSB},
    pixels::Color,
};

use self::player::Player;
use self::tone::Tone;

mod player;
mod tone;

const WIDTH: u32 = 800;
const HEIGHT: u32 = 600;

const PLAYBACK_SKIP: usize = (40 * FREQUENCY) as usize;
const FREQUENCY: i32 = 48000;
const FORMAT: u16 = AUDIO_F32LSB;
const CHANNELS: i32 = 2;
const CHUNK_SIZE: i32 = 1024;

const PIXELS_PER_FREQ: usize = 2;
const SCROLL_SPEED: usize = 10;

const SHOWN_VOLUME_MIN: i32 = -30;
const SHOWN_VOLUME_MAX: i32 = 150;
const SHOWN_VOLUME_RANGE: u32 = SHOWN_VOLUME_MIN.abs_diff(SHOWN_VOLUME_MAX);

const TIMECHART: bool = true;
const SPECTRUM: bool = true;

const SAMPLING_WINDOW_SEC: f64 = 10.0 / 60.0;
const IS_DB: bool = false;
const GAUSSIAN_SIGMA: f64 = 0.30;

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
        results.map(move |x| (x as f64) * val)
    };

    let fft_psd_db = if IS_DB {
        fft_data
            .into_iter()
            .map(|data| data.abs())
            .map(|as_| as_.powf(2.0))
            .map(|ps| ps / delta_f)
            .map(|psd| 10.0 * psd.log10())
            .collect::<Vec<_>>()
    } else {
        fft_data
            .into_iter()
            .map(|data| data.abs())
            .collect::<Vec<_>>()
    };

    fft_freq.zip(fft_psd_db).collect()
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

// https://python.atelierkobato.com/gaussian
// x: 0.0..1.0
fn gauss(x: f64) -> f64 {
    let a = 1.0;
    let mu = 0.0;
    let sigma = GAUSSIAN_SIGMA;
    a * (-((x * 2.0 - 1.0) - mu).powf(2.0) / (2.0 * sigma.powf(2.0))).exp()
}

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

    let mut history = (0..(HEIGHT as usize / SCROLL_SPEED))
        .map(|_| Vec::<(usize, f64)>::new())
        .collect::<VecDeque<_>>();

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

        render(
            &mut canvas,
            wavspec.sample_rate,
            began_at,
            &samples,
            &mut history,
        );
        std::thread::sleep(Duration::from_secs_f64(1.0 / 60.0));
    }
}

fn render(
    canvas: &mut sdl2::render::Canvas<sdl2::video::Window>,
    sample_rate: u32,
    began_at: Instant,
    samples: &[f64],
    history: &mut VecDeque<Vec<(usize, f64)>>,
) {
    canvas.set_draw_color(Color::BLACK);
    canvas.clear();

    let bps = sample_rate as f64;
    let rel_now = began_at.elapsed().as_secs_f64();

    let half_window = SAMPLING_WINDOW_SEC / 2.0;
    let wave = &samples[(bps * (rel_now - half_window).max(0.0)) as usize
        ..(bps * (rel_now + half_window)) as usize];
    let wave_len = wave.len();
    let wave = wave
        .iter()
        .enumerate()
        .map(|(i, &x)| x * gauss(i as f64 / wave_len as f64))
        .collect::<Vec<_>>();

    let fft = fft(&wave, sample_rate as usize);

    let mut freq_guideline = enum_iterator::first::<Tone>();
    let mut prev_vol = fft[0].1;
    let mut chistory = vec![];

    if TIMECHART {
        for (y, h) in history.iter().enumerate().rev() {
            if y == 0 {
                continue;
            }
            for &(x, v) in h.iter() {
                if v < 3.0 {
                    continue;
                }
                let rel_volume =
                    ((f64::clamp(v, SHOWN_VOLUME_MIN as f64, SHOWN_VOLUME_MAX as f64))
                        - SHOWN_VOLUME_MIN as f64)
                        / SHOWN_VOLUME_RANGE as f64;
                let c: Rgb<u8, LinearRgb> =
                    Hsv::<u8, LinearRgb>::new(Deg(255 - (232.0 * rel_volume) as u8), 127, 255)
                        .to_rgb();
                canvas.set_draw_color(Color::RGB(c.r, c.g, c.b));
                canvas
                    .draw_line(
                        (x as i32, ((y - 1) * SCROLL_SPEED) as i32),
                        (x as i32, (y * SCROLL_SPEED) as i32),
                    )
                    .unwrap();
            }
            if y * SCROLL_SPEED > HEIGHT as usize {
                break;
            }
        }
    }

    for (i, &(freq, volume)) in fft.iter().enumerate().skip(1) {
        while let Some(fg) = freq_guideline && freq > fg.freq() {
            freq_guideline = enum_iterator::next(&fg);
            chistory.push((i * PIXELS_PER_FREQ, volume));
        }

        let y = |volume: f64| {
            let rel_volume = (volume.clamp(SHOWN_VOLUME_MIN as f64, SHOWN_VOLUME_MAX as f64))
                + SHOWN_VOLUME_MIN.abs() as f64;
            let pos = rel_volume / SHOWN_VOLUME_RANGE as f64;
            (pos * HEIGHT as f64).clamp(1.0, HEIGHT as f64 - 1.0)
        };

        if SPECTRUM {
            canvas.set_draw_color(Color::WHITE);
            canvas
                .draw_line(
                    (
                        ((i - 1) * PIXELS_PER_FREQ) as i32,
                        HEIGHT as i32 - y(prev_vol) as i32,
                    ),
                    (
                        (i * PIXELS_PER_FREQ) as i32,
                        HEIGHT as i32 - y(volume) as i32,
                    ),
                )
                .unwrap();
            prev_vol = volume;
        }

        if i * PIXELS_PER_FREQ > WIDTH as usize {
            break;
        }
    }

    history.pop_front();
    history.push_back(chistory);

    canvas.present();
}

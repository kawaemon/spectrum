#![feature(let_chains)]

use std::{
    collections::VecDeque,
    ops::ControlFlow,
    time::{Duration, Instant},
};

use color::{color_space::LinearRgb, Deg, Hsv, Rgb, ToRgb};
use enum_iterator::Sequence;
use rustfft::{
    num_complex::{Complex, ComplexFloat},
    FftPlanner,
};
use sdl2::{
    audio::{AudioCallback, AudioFormat, AudioSpecDesired},
    event::Event,
    keyboard::Keycode,
    mixer::{InitFlag, AUDIO_F32LSB},
    pixels::Color,
};

const WIDTH: u32 = 800;
const HEIGHT: u32 = 600;

const PLAYBACK_SKIP: usize = (40 * FREQUENCY) as usize;
const FREQUENCY: i32 = 48000;
const FORMAT: u16 = AUDIO_F32LSB;
const CHANNELS: i32 = 2;
const CHUNK_SIZE: i32 = 1024;
const VOLUME: f32 = 0.5;

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
            struct Player {
                pos: usize,
                data: Vec<f64>,
            }
            impl AudioCallback for Player {
                type Channel = f32;
                fn callback(&mut self, x: &mut [f32]) {
                    assert_eq!(x.len(), 4096);
                    let wave = &self.data[self.pos..self.pos + x.len()];
                    for (i, x) in x.iter_mut().enumerate() {
                        *x = wave[i] as f32 * VOLUME;
                    }
                    self.pos += x.len();
                }
            }
            Player {
                pos: 0,
                data: played,
            }
        })
        .unwrap();

    let mut event_pump = sdl.event_pump().unwrap();
    device.resume();
    let began_at = Instant::now();

    std::thread::sleep(Duration::from_secs_f64(0.5));

    let mut history = (0..(HEIGHT as usize / SCROLL_SPEED))
        .map(|_| Vec::<(usize, f64)>::new())
        .collect::<VecDeque<_>>();

    loop {
        if let ControlFlow::Break(_) = render(
            &mut event_pump,
            &mut canvas,
            wavspec,
            began_at,
            &samples,
            &mut history,
        ) {
            break;
        }
        std::thread::sleep(Duration::from_secs_f64(1.0 / 60.0));
    }
}

fn render(
    event_pump: &mut sdl2::EventPump,
    canvas: &mut sdl2::render::Canvas<sdl2::video::Window>,
    wavspec: hound::WavSpec,
    began_at: Instant,
    samples: &Vec<f64>,
    history: &mut VecDeque<Vec<(usize, f64)>>,
) -> ControlFlow<()> {
    for event in event_pump.poll_iter() {
        match event {
            Event::Quit { .. }
            | Event::KeyDown {
                keycode: Some(Keycode::Q | Keycode::Escape),
                ..
            } => return ControlFlow::Break(()),
            _ => {}
        }
    }

    canvas.set_draw_color(Color::BLACK);
    canvas.clear();

    let bps = wavspec.sample_rate as f64;
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

    let fft = fft(&wave, wavspec.sample_rate as usize);

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

    ControlFlow::Continue(())
}

#[derive(Debug, Clone, Copy, Sequence)]
enum Tone {
    // A0,
    // As0,
    // B0,
    // C1,
    // Cs1,
    // D1,
    // Ds1,
    // E1,
    // F1,
    // Fs1,
    // G1,
    // Gs1,
    // A1,
    // As1,
    // B1,
    // C2,
    // Cs2,
    // D2,
    // Ds2,
    // E2,
    // F2,
    // Fs2,
    // G2,
    // Gs2,
    // A2,
    // As2,
    // B2,
    C3,
    Cs3,
    D3,
    Ds3,
    E3,
    F3,
    Fs3,
    G3,
    Gs3,
    A3,
    As3,
    B3,
    C4,
    Cs4,
    D4,
    Ds4,
    E4,
    F4,
    Fs4,
    G4,
    Gs4,
    A4,
    As4,
    B4,
    C5,
    Cs5,
    D5,
    Ds5,
    E5,
    F5,
    Fs5,
    G5,
    Gs5,
    A5,
    As5,
    B5,
    C6,
    Cs6,
    D6,
    Ds6,
    E6,
    F6,
    Fs6,
    G6,
    Gs6,
    A6,
    As6,
    B6,
    // C7,
    // Cs7,
    // D7,
    // Ds7,
    // E7,
    // F7,
    // Fs7,
    // G7,
    // Gs7,
    // A7,
    // As7,
    // B7,
    // C8,
}

impl Tone {
    fn freq(&self) -> f64 {
        match self {
            // Tone::A0 => 27.500,
            // Tone::As0 => 29.135,
            // Tone::B0 => 30.868,
            // Tone::C1 => 32.703,
            // Tone::Cs1 => 34.648,
            // Tone::D1 => 36.708,
            // Tone::Ds1 => 38.891,
            // Tone::E1 => 41.203,
            // Tone::F1 => 43.654,
            // Tone::Fs1 => 46.249,
            // Tone::G1 => 48.999,
            // Tone::Gs1 => 51.913,
            // Tone::A1 => 55.000,
            // Tone::As1 => 58.270,
            // Tone::B1 => 61.735,
            // Tone::C2 => 65.406,
            // Tone::Cs2 => 69.296,
            // Tone::D2 => 73.416,
            // Tone::Ds2 => 77.782,
            // Tone::E2 => 82.407,
            // Tone::F2 => 87.307,
            // Tone::Fs2 => 92.499,
            // Tone::G2 => 97.999,
            // Tone::Gs2 => 103.82,
            // Tone::A2 => 110.00,
            // Tone::As2 => 116.54,
            // Tone::B2 => 123.47,
            Tone::C3 => 130.81,
            Tone::Cs3 => 138.59,
            Tone::D3 => 146.83,
            Tone::Ds3 => 155.56,
            Tone::E3 => 164.81,
            Tone::F3 => 174.61,
            Tone::Fs3 => 184.99,
            Tone::G3 => 195.99,
            Tone::Gs3 => 207.65,
            Tone::A3 => 220.00,
            Tone::As3 => 233.08,
            Tone::B3 => 246.94,
            Tone::C4 => 261.62,
            Tone::Cs4 => 277.18,
            Tone::D4 => 293.66,
            Tone::Ds4 => 311.12,
            Tone::E4 => 329.62,
            Tone::F4 => 349.22,
            Tone::Fs4 => 369.99,
            Tone::G4 => 391.99,
            Tone::Gs4 => 415.30,
            Tone::A4 => 440.00,
            Tone::As4 => 466.16,
            Tone::B4 => 493.88,
            Tone::C5 => 523.25,
            Tone::Cs5 => 554.36,
            Tone::D5 => 587.33,
            Tone::Ds5 => 622.25,
            Tone::E5 => 659.25,
            Tone::F5 => 698.45,
            Tone::Fs5 => 739.98,
            Tone::G5 => 783.99,
            Tone::Gs5 => 830.60,
            Tone::A5 => 880.00,
            Tone::As5 => 932.32,
            Tone::B5 => 987.76,
            Tone::C6 => 1046.502,
            Tone::Cs6 => 1108.731,
            Tone::D6 => 1174.659,
            Tone::Ds6 => 1244.508,
            Tone::E6 => 1318.510,
            Tone::F6 => 1396.913,
            Tone::Fs6 => 1479.978,
            Tone::G6 => 1567.982,
            Tone::Gs6 => 1661.219,
            Tone::A6 => 1760.000,
            Tone::As6 => 1864.655,
            Tone::B6 => 1975.533,
            // Tone::C7 => 2093.005,
            // Tone::Cs7 => 2217.461,
            // Tone::D7 => 2349.318,
            // Tone::Ds7 => 2489.016,
            // Tone::E7 => 2637.020,
            // Tone::F7 => 2793.826,
            // Tone::Fs7 => 2959.955,
            // Tone::G7 => 3135.963,
            // Tone::Gs7 => 3322.438,
            // Tone::A7 => 3520.000,
            // Tone::As7 => 3729.310,
            // Tone::B7 => 3951.066,
            // Tone::C8 => 4186.009,
        }
    }
}

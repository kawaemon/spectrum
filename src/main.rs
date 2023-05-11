#![feature(let_chains)]

use std::{
    collections::VecDeque,
    time::{Duration, Instant},
};

use color::{color_space::LinearRgb, Deg, Hsv, Rgb, ToRgb};
use enum_iterator::Sequence;
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

const WIDTH: u32 = 1800;
const HEIGHT: u32 = 900;

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
        results.map(move |x| (x as f64) * val)
    };

    let fft_psd_db = fft_data
        .into_iter()
        .map(|data| data.abs())
        //.map(|as_| as_.powf(2.0))
        //.map(|ps| ps / delta_f)
        // .map(|psd| 10.0 * psd.log10())
        .map(|x| x / 10.0)
        .collect::<Vec<_>>();

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
    let sigma = 0.15;
    a * (-((x * 2.0 - 1.0) - mu).powf(2.0) / (2.0 * sigma.powf(2.0))).exp()
}

fn main() {
    let sdl = sdl2::init().unwrap();
    let video = sdl.video().unwrap();
    let _audio = sdl.audio().unwrap();

    sdl2::mixer::open_audio(FREQUENCY, FORMAT, CHANNELS, CHUNK_SIZE).unwrap();
    let _mixer_context = sdl2::mixer::init(InitFlag::MP3).unwrap();

    let sdl_source = Music::from_file("./out.mp3").unwrap();

    let wav = hound::WavReader::open("./out.wav").unwrap();
    let wavspec = wav.spec();
    assert_eq!(wavspec.channels, 2);

    let (left_samples, _right_samples, _) = wav
        .into_samples::<f32>()
        .try_fold(
            (vec![], vec![], true),
            |(mut left, mut right, is_left), v| {
                let v = v? as f64;
                if is_left {
                    left.push(v);
                } else {
                    right.push(v);
                }
                Result::<_, hound::Error>::Ok((left, right, !is_left))
            },
        )
        .unwrap();

    let window = video
        .window("spectrum", WIDTH, HEIGHT)
        .position_centered()
        .opengl()
        .build()
        .unwrap();

    let mut canvas = window.into_canvas().build().unwrap();

    let mut event_pump = sdl.event_pump().unwrap();

    sdl_source.play(0).unwrap();
    let began_at = Instant::now();

    std::thread::sleep(Duration::from_secs_f64(0.5));

    const SCROLL_SPEED: usize = 10;

    let mut history = (0..(HEIGHT as usize / SCROLL_SPEED))
        .map(|_| Vec::<(usize, f64)>::new())
        .collect::<VecDeque<_>>();

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

        canvas.set_draw_color(Color::BLACK);
        canvas.clear();

        const SHOWN_FREQ_MAX: usize = 5_000;
        const SAMPLING_WINDOW_SEC: f64 = 15.0 / 60.0;

        let bps = wavspec.sample_rate as f64;
        let rel_now = began_at.elapsed().as_secs_f64();

        let half_window = SAMPLING_WINDOW_SEC / 2.0;
        let wave = &left_samples[(bps * (rel_now - half_window).max(0.0)) as usize
            ..(bps * (rel_now + half_window)) as usize];
        let wave_len = wave.len();
        let wave = wave
            .iter()
            .enumerate()
            .map(|(i, &x)| x * gauss(i as f64 / wave_len as f64))
            .collect::<Vec<_>>();

        let fft = fft(&wave, wavspec.sample_rate as usize);
        let pos = fft
            .iter()
            .position(|x| x.0 > SHOWN_FREQ_MAX as f64)
            .unwrap();

        let mut freq_guideline = enum_iterator::first::<Tone>();
        let mut prev_vol = fft[0].1;
        let mut chistory = vec![];

        const SHOWN_VOLUME_MIN: i32 = 0;
        const SHOWN_VOLUME_MAX: i32 = 100;
        const SHOWN_VOLUME_RANGE: u32 = SHOWN_VOLUME_MIN.abs_diff(SHOWN_VOLUME_MAX);
        const PIXELS_PER_FREQ: usize = 5;

        for (y, h) in history.iter().enumerate().rev() {
            if y == 0 {
                continue;
            }
            for &(x, v) in h.iter() {
                if v < 10.0 {
                    continue;
                }
                let rel_volume =
                    ((f64::clamp(v, SHOWN_VOLUME_MIN as f64, SHOWN_VOLUME_MAX as f64))
                        + SHOWN_VOLUME_MIN.abs() as f64)
                        / SHOWN_VOLUME_RANGE as f64;
                let c: Rgb<u8, LinearRgb> = Hsv::<u8, LinearRgb>::new(
                    Deg(232),
                    ((1.0 - rel_volume * 1.5) * 255.0).min(254.0) as u8,
                    255,
                )
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

        for (i, &(freq, mut volume)) in fft.iter().enumerate().skip(1) {
            volume *= 5.0;
            volume -= 20.0;

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

            if i * PIXELS_PER_FREQ > WIDTH as usize {
                break;
            }
        }

        history.pop_front();
        history.push_back(chistory);

        canvas.present();

        std::thread::sleep(Duration::from_secs_f64(1.0 / 60.0));
    }
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

/*
27.500       A0
29.135      As0
30.868       B0
32.703       C1
34.648      Cs1
36.708       D1
38.891      Ds1
41.203       E1
 43.654      F1
 46.249     Fs1
 48.999       G1
 51.913     Gs1
 55.000     A1
 58.270     As1
 61.735    B1
 65.406     C2
 69.296     Cs2
 73.416     D2
 77.782     Ds2
 82.407     E2
 87.307     F2
 92.499     Fs2
 97.999     G2
 103.82    Gs2
 110.00     A2
 116.54     As2
 123.47     B2
 130.81     C3
 138.59     Cs3
 146.83     D3
 155.56     Ds3
 164.81     E3
 174.61     F3
 184.99     Fs3
 195.99     G3
 207.65     Gs3
 220.00     A3
 233.08     As3
 246.94     B3
 261.62     C4
 277.18     Cs4
 293.66     D4
 311.12     Ds4
 329.62     E4
 349.22     F4
 369.99     Fs4
 391.99     G4
 415.30     Gs4
 440.00     A4
 466.16     As4
 493.88     B4
 523.25     C5
 554.36     Cs5
 587.33     D5
 622.25     Ds5
 659.25     E5
 698.45     F5
 739.98     Fs5
 783.99     G5
 830.60     Gs5
 880.00     A5
 932.32     As5
 987.76     B5
 1046.502   C6
 1108.731   Cs6
 1174.659   D6
 1244.508   Ds6
 1318.510   E6
 1396.913   F6
 1479.978   Fs6
 1567.982   G6
 1661.219   Gs6
 1760.000   A6
 1864.655   As6
 1975.533   B6
 2093.005   C7
 2217.461   Cs7
 2349.318   D7
 2489.016   Ds7
 2637.020   E7
 2793.826   F7
 2959.955   Fs7
 3135.963   G7
 3322.438   Gs7
 3520.000   A7
 3729.310   As7
 3951.066   B7
 4186.009   C8



 */

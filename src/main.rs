use std::time::{Duration, Instant};

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

const WIDTH: u32 = 960;
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
        results.map(move |x| (x as f64) * val)
    };

    let fft_psd_db = fft_data
        .into_iter()
        .map(|data| data.abs())
        // .map(|as_| as_.powf(2.0))
        // .map(|ps| ps / delta_f)
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
    let sigma = 0.25;
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
        const FREQ_GUIDELINE_DISTANCE: usize = 100;
        const SAMPLING_WINDOW_SEC: f64 = 10.0 / 60.0;

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

        let mut prev_freq = fft[0].0;
        let mut prev_vol = fft[0].1;

        for (i, fft_index) in (0..pos).enumerate().skip(1) {
            let (freq, volume) = fft[fft_index];

            if (freq - prev_freq) > FREQ_GUIDELINE_DISTANCE as f64 {
                prev_freq = freq;
                canvas.set_draw_color(Color::RGB(0, 143, 0));
                canvas.draw_line((i as i32, 92), (i as i32, 100)).unwrap();
            }

            const SHOWN_VOLUME_MIN: i32 = 0;
            const SHOWN_VOLUME_MAX: i32 = 100;
            const SHOWN_VOLUME_RANGE: u32 = SHOWN_VOLUME_MIN.abs_diff(SHOWN_VOLUME_MAX);

            let y = |volume: f64| {
                let rel_volume = (volume.clamp(SHOWN_VOLUME_MIN as f64, SHOWN_VOLUME_MAX as f64))
                    + SHOWN_VOLUME_MIN.abs() as f64;
                let pos = rel_volume / SHOWN_VOLUME_RANGE as f64;
                (pos * HEIGHT as f64).clamp(1.0, HEIGHT as f64 - 1.0)
            };

            canvas.set_draw_color(Color::WHITE);
            canvas
                .draw_line(
                    ((i - 1) as i32, HEIGHT as i32 - y(prev_vol) as i32),
                    (i as i32, HEIGHT as i32 - y(volume) as i32),
                )
                .unwrap();
            prev_vol = volume;
        }

        canvas.present();

        std::thread::sleep(Duration::from_secs_f64(1.0 / 60.0));
    }
}

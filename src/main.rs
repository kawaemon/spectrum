use std::time::{Duration, Instant};

use rustfft::{
    num_complex::{Complex, ComplexFloat},
    Fft, FftPlanner,
};
use sdl2::{
    audio::{AudioCVT, AudioCallback, AudioFormat, AudioSpecDesired, AudioSpecWAV},
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

fn ifft(mut freqs: &[f64], output_len: usize) -> Vec<f64> {
    if freqs.len() > output_len {
        freqs = &freqs[0..output_len];
    }
    let mut freqs = freqs
        .iter()
        .copied()
        .chain(std::iter::repeat(0.0).take(output_len.saturating_sub(freqs.len())))
        .map(|x| Complex { re: x, im: 0.0 })
        .collect::<Vec<_>>();
    assert_eq!(freqs.len(), output_len);
    FftPlanner::new()
        .plan_fft_inverse(freqs.len())
        .process(&mut freqs);
    freqs.into_iter().map(|x| x.abs()).collect()
}

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

    // let fft_data = fft_data.into_iter().map(|x| x.abs()).collect::<Vec<_>>();

    // let min = fft_data.iter().min_by(|a, b| a.total_cmp(b)).unwrap();
    // // let max = fft_data.iter().max_by(|a, b| a.total_cmp(b)).unwrap();
    // let max = 800.0;

    // let normalized = fft_data.iter().map(|x| x / (max - min));

    // fft_freq.zip(normalized).collect()

    let fft_psd_db = fft_data
        .into_iter()
        .map(|data| data.abs())
        .map(|as_| as_.powf(2.0))
        .map(|ps| ps / delta_f)
        .map(|psd| 10.0 * psd.log10())
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

// https://www.utsbox.com/?page_id=523
fn low_pass() -> (f64, f64, f64, f64, f64, f64) {
    let samplerate = 48000.0;
    let freq = 2400.0;
    let q = 1.0 / (2.0.sqrt());
    let omega = std::f64::consts::PI * 2.0 * freq / samplerate;
    let alpha = omega.sin() / (2.0 * q);
    let a0 = 1.0 + alpha;
    let a1 = -2.0 * omega.cos();
    let a2 = 1.0 - alpha;
    let b0 = (1.0 - omega.cos()) / 2.0;
    let b1 = 1.0 - omega.cos();
    let b2 = (1.0 - omega.cos()) / 2.0;
    (a0, a1, a2, b0, b1, b2)
}

fn high_pass() -> (f64, f64, f64, f64, f64, f64) {
    let samplerate = 48000.0;
    let freq = 2400.0;
    let q = 1.0 / (2.0.sqrt());
    let omega = std::f64::consts::PI * 2.0 * freq / samplerate;
    let alpha = omega.sin() / (2.0 * q);
    let a0 = 1.0 + alpha;
    let a1 = -2.0 * omega.cos();
    let a2 = 1.0 - alpha;
    let b0 = (1.0 + omega.cos()) / 2.0;
    let b1 = -(1.0 + omega.cos());
    let b2 = (1.0 + omega.cos()) / 2.0;
    (a0, a1, a2, b0, b1, b2)
}

fn band_pass() -> (f64, f64, f64, f64, f64, f64) {
    let samplerate = 48000.0;
    let freq = 440.0;
    let bw = 0.3;
    let omega = 2.0 * std::f64::consts::PI * freq / samplerate;
    let alpha =
        omega.sin() * (2.0.log(std::f64::consts::E) / 2.0 * bw * omega / omega.sin()).sinh();
    let a0 = 1.0 + alpha;
    let a1 = -2.0 * omega.cos();
    let a2 = 1.0 - alpha;
    let b0 = alpha;
    let b1 = 0.0;
    let b2 = -alpha;
    (a0, a1, a2, b0, b1, b2)
}

fn peek() -> (f64, f64, f64, f64, f64, f64) {
    let samplerate = 48000.0;
    let freq = 2400.0;
    let gain = 15.0;
    let bw = 0.3;
    let omega = 2.0 * std::f64::consts::PI * freq / samplerate;
    let alpha =
        omega.sin() * (2.0.log(std::f64::consts::E) / 2.0 * bw * omega / omega.sin()).sinh();
    let a = 10.0.powf(gain / 40.0);

    let a0 = 1.0 + alpha / a;
    let a1 = -2.0 * omega.cos();
    let a2 = 1.0 - alpha / a;
    let b0 = 1.0 + alpha * a;
    let b1 = -2.0 * omega.cos();
    let b2 = 1.0 - alpha * a;
    (a0, a1, a2, b0, b1, b2)
}

fn convert(wav: &[f64]) -> Vec<f64> {
    return wav.to_vec();
    let mut output = Vec::with_capacity(wav.len());

    let (a0, a1, a2, b0, b1, b2) = band_pass();
    let mut in2 = 0.0;
    let mut in1 = 0.0;
    let mut out1 = 0.0;
    let mut out2 = 0.0;

    for i in 0..wav.len() {
        output.push(
            b0 / a0 * wav[i] + b1 / a0 * in1 + b2 / a0 * in2 - a1 / a0 * out1 - a2 / a0 * out2,
        );

        in2 = in1;
        in1 = wav[i];

        out2 = out1;
        out1 = output[i];
    }

    output

    // for(int i = 0; i < size; i++)
    // {
    // 	// 入力信号にフィルタを適用し、出力信号として書き出す。
    // output[i] =
    //     b0 / a0 * input[i] + b1 / a0 * in1 + b2 / a0 * in2 - a1 / a0 * out1 - a2 / a0 * out2;
    //
    // 	in2  = in1;       // 2つ前の入力信号を更新
    // 	in1  = input[i];  // 1つ前の入力信号を更新
    //
    // 	out2 = out1;      // 2つ前の出力信号を更新
    // 	out1 = output[i]; // 1つ前の出力信号を更新
    // }
}

fn main() {
    let sdl = sdl2::init().unwrap();
    let video = sdl.video().unwrap();
    let audio = sdl.audio().unwrap();

    sdl2::mixer::open_audio(FREQUENCY, FORMAT, CHANNELS, CHUNK_SIZE).unwrap();
    let _mixer_context = sdl2::mixer::init(InitFlag::MP3).unwrap();

    let wav = hound::WavReader::open("./a.wav").unwrap();
    let wavspec = wav.spec();
    assert_eq!(wavspec.channels, 2);
    assert_eq!(wavspec.sample_rate, 48000);

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

    let shown_samples = left_samples.clone();

    let window = video
        .window("spectrum", WIDTH, HEIGHT)
        .position_centered()
        .opengl()
        .build()
        .unwrap();

    let desired_spec = AudioSpecDesired {
        freq: Some(FREQUENCY),
        channels: Some(1),
        samples: None,
    };

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
                    let wave = convert(wave);
                    for (i, x) in x.iter_mut().enumerate() {
                        *x = wave[i] as f32;
                    }
                    self.pos += x.len();
                }
            }
            Player {
                pos: 0,
                data: left_samples,
            }
        })
        .unwrap();

    let mut canvas = window.into_canvas().build().unwrap();
    let mut event_pump = sdl.event_pump().unwrap();

    device.resume();
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
        const SAMPLING_WINDOW_SEC: f64 = 6.0 / 60.0;

        let bps = wavspec.sample_rate as f64;
        let rel_now = began_at.elapsed().as_secs_f64();

        let half_window = SAMPLING_WINDOW_SEC / 2.0;
        let wave = &shown_samples[(bps * (rel_now - half_window).max(0.0)) as usize
            ..(bps * (rel_now + half_window)) as usize];
        let wave = convert(wave);
        let fft = fft(&wave, wavspec.sample_rate as usize);
        let fft = fft
            .into_iter()
            .filter(|x| x.0 <= SHOWN_FREQ_MAX as f64)
            .collect::<Vec<_>>();
        let pos = fft.len();

        let mut prev_freq = fft[0].0;
        let mut prev_volume = fft[0].1;

        for (i, fft_index) in (1..pos).enumerate() {
            let (freq, volume) = fft[fft_index];

            if (freq - prev_freq) > FREQ_GUIDELINE_DISTANCE as f64 {
                prev_freq = freq;
                canvas.set_draw_color(Color::RGB(0, 143, 0));
                canvas
                    .draw_line((i as i32 * 2, 92), (i as i32 * 2, 100))
                    .unwrap();
            }

            const SHOWN_VOLUME_MIN: i32 = -50;
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
                    ((i as i32 * 2 - 1), (HEIGHT as f64 - y(prev_volume)) as i32),
                    ((i as i32 * 2), (HEIGHT as f64 - y(volume)) as i32),
                )
                .unwrap();
            prev_volume = volume;
        }

        canvas.present();

        std::thread::sleep(Duration::from_secs_f64(1.0 / 60.0));
    }
}

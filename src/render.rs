use std::{collections::VecDeque, time::Instant};

use color::{color_space::LinearRgb, Deg, Hsv, Rgb, ToRgb};
use sdl2::pixels::Color;

use crate::{fft::fft, tone::Tone, HEIGHT, WIDTH};

const PIXELS_PER_FREQ: usize = 2;
const SCROLL_SPEED: usize = 10;

const SHOWN_VOLUME_MIN: i32 = -30;
const SHOWN_VOLUME_MAX: i32 = 150;
const SHOWN_VOLUME_RANGE: u32 = SHOWN_VOLUME_MIN.abs_diff(SHOWN_VOLUME_MAX);

pub struct Renderer {
    history: VecDeque<Vec<(usize, f64)>>,
}

impl Renderer {
    pub fn new() -> Self {
        Self {
            history: (0..(HEIGHT as usize / SCROLL_SPEED))
                .map(|_| Vec::<(usize, f64)>::new())
                .collect::<VecDeque<_>>(),
        }
    }

    pub fn render(
        &mut self,
        canvas: &mut sdl2::render::Canvas<sdl2::video::Window>,
        sample_rate: u32,
        began_at: Instant,
        samples: &[f64],
    ) {
        canvas.set_draw_color(Color::BLACK);
        canvas.clear();

        let bps = sample_rate as f64;
        let rel_now = began_at.elapsed().as_secs_f64();

        const SAMPLING_WINDOW_SEC: f64 = 10.0 / 60.0;
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

        if cfg!(feature = "time_chart") {
            for (y, h) in self.history.iter().enumerate().rev() {
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
            while freq_guideline.map_or(false, |fg| fg.freq() < freq) {
                freq_guideline = enum_iterator::next(&freq_guideline.unwrap());
                chistory.push((i * PIXELS_PER_FREQ, volume));
            }

            let y = |volume: f64| {
                let rel_volume = (volume.clamp(SHOWN_VOLUME_MIN as f64, SHOWN_VOLUME_MAX as f64))
                    + SHOWN_VOLUME_MIN.abs() as f64;
                let pos = rel_volume / SHOWN_VOLUME_RANGE as f64;
                (pos * HEIGHT as f64).clamp(1.0, HEIGHT as f64 - 1.0)
            };

            if cfg!(feature = "spectrum") {
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

        self.history.pop_front();
        self.history.push_back(chistory);

        canvas.present();
    }
}

// https://python.atelierkobato.com/gaussian
// x: 0.0..1.0
fn gauss(x: f64) -> f64 {
    const GAUSSIAN_SIGMA: f64 = 0.30;
    let a = 1.0;
    let mu = 0.0;
    let sigma = GAUSSIAN_SIGMA;
    a * (-((x * 2.0 - 1.0) - mu).powf(2.0) / (2.0 * sigma.powf(2.0))).exp()
}

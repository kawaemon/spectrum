use std::{collections::VecDeque, time::Instant};

use color::{color_space::LinearRgb, Deg, Hsv, Rgb, ToRgb};
use sdl2::{pixels::Color, rect::Rect};

use crate::{fft::fft, tone::Key, HEIGHT, WIDTH};

const PIXELS_PER_FREQ: usize = 2;
const SCROLL_SPEED: usize = 10;
const KEY_WIDTH: u32 = 12;
const KEY_HEIGHT: u32 = 100;

pub struct Renderer {
    history: VecDeque<Vec<(usize, f64)>>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct RenderProps {
    pub show_time_chart: bool,
    pub show_spectrum: bool,
}

impl Default for RenderProps {
    fn default() -> Self {
        Self {
            show_time_chart: true,
            show_spectrum: true,
        }
    }
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
        props: &RenderProps,
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
        let wave_start = (bps * (rel_now - half_window).max(0.0)) as usize;
        let wave = if wave_start < samples.len() {
            &samples[wave_start..samples.len().min((bps * (rel_now + half_window)) as usize)]
        } else {
            &[]
        };
        let wave_len = wave.len();
        let wave = wave
            .iter()
            .enumerate()
            .map(|(i, &x)| x * gauss(i as f64 / wave_len as f64))
            .collect::<Vec<_>>();

        let fft = fft(&wave, sample_rate as usize);

        if props.show_time_chart {
            for key in enum_iterator::all::<Key>() {
                if key.tone().is_white() {
                    canvas.set_draw_color(Color::WHITE);
                } else {
                    canvas.set_draw_color(Color::BLACK);
                }
                canvas
                    .fill_rect(Rect::new(
                        key.position() as i32 * KEY_WIDTH as i32,
                        HEIGHT as i32 - KEY_HEIGHT as i32,
                        KEY_WIDTH - 1,
                        KEY_HEIGHT,
                    ))
                    .unwrap();
            }
            for (y, h) in self.history.iter().enumerate().skip(1).rev() {
                for &(x, v) in h.iter() {
                    if v < 3.0 {
                        continue;
                    }
                    let rel_volume = normalize_volume(v);
                    let c: Rgb<u8, LinearRgb> =
                        Hsv::<u8, LinearRgb>::new(Deg(255 - (232.0 * rel_volume) as u8), 127, 255)
                            .to_rgb();
                    canvas.set_draw_color(Color::RGB(c.r, c.g, c.b));
                    let line_x = (x * PIXELS_PER_FREQ) as i32;
                    canvas
                        .draw_line(
                            (line_x, ((y - 1) * SCROLL_SPEED) as i32),
                            (line_x, (y * SCROLL_SPEED) as i32),
                        )
                        .unwrap();
                }
                if y * SCROLL_SPEED > HEIGHT as usize {
                    break;
                }
            }
        }

        let mut current_history = vec![];
        let mut freq_guideline = enum_iterator::first::<Key>();
        for (i, window) in fft.windows(2).enumerate().skip(1) {
            let (freq, volume) = window[0];
            while freq_guideline.map_or(false, |fg| fg.freq() < freq) {
                freq_guideline = enum_iterator::next(&freq_guideline.unwrap());
                current_history.push((i, volume));
            }

            if props.show_spectrum {
                fn envelope_y(volume: f64) -> f64 {
                    let pos = normalize_volume(volume);
                    (pos * HEIGHT as f64).clamp(1.0, HEIGHT as f64 - 1.0)
                }
                canvas.set_draw_color(Color::WHITE);
                canvas
                    .draw_line(
                        (
                            ((i - 1) * PIXELS_PER_FREQ) as i32,
                            HEIGHT as i32 - envelope_y(volume) as i32,
                        ),
                        (
                            (i * PIXELS_PER_FREQ) as i32,
                            HEIGHT as i32 - envelope_y(window[1].1) as i32,
                        ),
                    )
                    .unwrap();
            }

            if i * PIXELS_PER_FREQ > WIDTH as usize {
                break;
            }
        }
        self.history.pop_front();
        self.history.push_back(current_history);

        canvas.present();
    }
}

fn normalize_volume(volume: f64) -> f64 {
    const SHOWN_VOLUME_MIN: i32 = -30;
    const SHOWN_VOLUME_MAX: i32 = 150;
    const SHOWN_VOLUME_RANGE: u32 = SHOWN_VOLUME_MIN.abs_diff(SHOWN_VOLUME_MAX);

    ((f64::clamp(volume, SHOWN_VOLUME_MIN as f64, SHOWN_VOLUME_MAX as f64))
        - SHOWN_VOLUME_MIN as f64)
        / SHOWN_VOLUME_RANGE as f64
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

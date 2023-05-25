use sdl2::audio::AudioCallback;

const VOLUME: f32 = 0.5;

#[derive(Debug)]
pub struct Player {
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

impl Player {
    pub fn new(data: Vec<f64>) -> Self {
        Self { pos: 0, data }
    }
}

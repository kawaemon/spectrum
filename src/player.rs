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
        let wave = if self.pos < self.data.len() {
            &self.data[self.pos..self.data.len().min(self.pos + x.len())]
        } else {
            &[]
        };
        x.fill(0.0);
        for (x, &sample) in x.iter_mut().zip(wave) {
            *x = sample as f32 * VOLUME;
        }
        self.pos += x.len();
    }
}

impl Player {
    pub fn new(data: Vec<f64>) -> Self {
        Self { pos: 0, data }
    }
}

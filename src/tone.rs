use enum_iterator::Sequence;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Sequence)]
pub enum Tone {
    C,
    Cs,
    D,
    Ds,
    E,
    F,
    Fs,
    G,
    Gs,
    A,
    As,
    B,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct Key {
    tone: Tone,
    octave: u8,
}

impl Key {
    pub fn new(tone: Tone, octave: u8) -> Self {
        assert!(octave < 64, "octave must be less than 64");
        Self { tone, octave }
    }

    pub const fn freq(&self) -> f64 {
        (match self.tone {
            Tone::A => 27.500,
            Tone::As => 29.135,
            Tone::B => 30.868,
            Tone::C => 32.703,
            Tone::Cs => 34.648,
            Tone::D => 36.708,
            Tone::Ds => 38.891,
            Tone::E => 41.203,
            Tone::F => 43.654,
            Tone::Fs => 46.249,
            Tone::G => 48.999,
            Tone::Gs => 51.913,
        }) * (1u64 << self.octave) as f64
    }
}

impl Sequence for Key {
    const CARDINALITY: usize = Tone::CARDINALITY * u8::MAX as usize;

    fn next(&self) -> Option<Self> {
        if let Tone::Gs = self.tone {
            if self.octave == u8::MAX {
                None
            } else {
                Some(Self {
                    tone: Tone::A,
                    octave: self.octave + 1,
                })
            }
        } else {
            Some(Self {
                tone: self.tone.next()?,
                ..*self
            })
        }
    }

    fn previous(&self) -> Option<Self> {
        if let Tone::Gs = self.tone {
            if self.octave == 0 {
                None
            } else {
                Some(Self {
                    tone: Tone::A,
                    octave: self.octave - 1,
                })
            }
        } else {
            Some(Self {
                tone: self.tone.previous()?,
                ..*self
            })
        }
    }

    fn first() -> Option<Self> {
        Some(Self {
            octave: 0,
            tone: Tone::A,
        })
    }

    fn last() -> Option<Self> {
        Some(Self {
            octave: 63,
            tone: Tone::Gs,
        })
    }
}

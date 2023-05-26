use enum_iterator::Sequence;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Sequence)]
pub enum Tone {
    A,
    As,
    B,
    C,
    Cs,
    D,
    Ds,
    E,
    F,
    Fs,
    G,
    Gs,
}

impl Tone {
    pub fn is_white(self) -> bool {
        use Tone::*;
        matches!(self, A | B | C | D | E | F | G)
    }

    pub fn is_black(self) -> bool {
        use Tone::*;
        matches!(self, As | Cs | Ds | Fs | Gs)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Key {
    tone: Tone,
    octave: u8,
}

impl Key {
    pub fn freq(&self) -> f64 {
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

    pub fn tone(&self) -> Tone {
        self.tone
    }

    /// Finds absolute X position of the key as below:
    ///
    /// ```text
    ///    1       4   6       9   11  13
    ///   [A#]    [C#][D#]    [F#][G#][A#]
    /// [A ][B ][C ][D ][E ][F ][G ][A ]  ...
    ///  0   2   3   5   7   8   10  12
    /// ```
    pub fn position(&self) -> u32 {
        (match self.tone {
            Tone::A => 0,
            Tone::As => 1,
            Tone::B => 2,
            Tone::C => 3,
            Tone::Cs => 4,
            Tone::D => 5,
            Tone::Ds => 6,
            Tone::E => 7,
            Tone::F => 8,
            Tone::Fs => 9,
            Tone::G => 10,
            Tone::Gs => 11,
        }) + self.octave as u32 * 12
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

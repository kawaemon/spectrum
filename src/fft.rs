use rustfft::{
    num_complex::{Complex, ComplexFloat},
    FftPlanner,
};

// https://ryo-iijima.com/fftresult/
pub fn fft(signal: &[f64], sampling_freq: usize) -> Vec<(f64, f64)> {
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

    let fft_psd_db = if cfg!(feature = "db") {
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

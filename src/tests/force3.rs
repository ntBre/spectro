use approx::assert_abs_diff_eq;

use crate::*;

#[derive(Clone)]
struct Test {
    infile: String,
    fort15: String,
    fort30: String,
    want: Vec<f64>,
    eps: f64,
}

impl Test {
    fn new(dir: &'static str, eps: f64) -> Self {
        let start = Path::new("testfiles");
        Self {
            infile: String::from(
                start.join(dir).join("spectro.in").to_str().unwrap(),
            ),
            fort15: String::from(
                start.join(dir).join("fort.15").to_str().unwrap(),
            ),
            fort30: String::from(
                start.join(dir).join("fort.30").to_str().unwrap(),
            ),
            want: load_vec(start.join(dir).join("f3qcm")),
            eps,
        }
    }
}

#[test]
pub(crate) fn test_force3() {
    let tests = [
        Test::new("h2o", 4e-6),
        Test::new("h2co", 3e-6),
        Test::new("c3h2", 7e-6),
        Test::new("c3hf", 5e-6),
        Test::new("c3hcn", 5e-6),
    ];
    for test in tests {
        let s = Spectro::load(&test.infile);
        let fc2 = load_fc2(&test.fort15, s.n3n);
        let fc2 = s.rot2nd(fc2);
        let fc2 = FACT2 * fc2;
        let w = s.geom.weights();
        let sqm: Vec<_> = w.iter().map(|w| 1.0 / w.sqrt()).collect();
        let fxm = s.form_sec(fc2, &sqm);
        let (harms, lxm) = symm_eigen_decomp(fxm);
        let freq = to_wavenumbers(&harms);
        let lx = s.make_lx(&sqm, &lxm);
        let f3x = load_fc3(&test.fort30, s.n3n);
        let mut f3x = s.rot3rd(f3x, s.axes);
        let got = force3(s.n3n, &mut f3x, &lx, s.nvib, &freq);
        let got = Dvec::from(got).abs();
        let want = Dvec::from(test.want).abs();
        // println!("\ndiff = {:.2e}", (got.clone() - want.clone()).max());
        assert_abs_diff_eq!(got, want, epsilon = test.eps);
    }
}

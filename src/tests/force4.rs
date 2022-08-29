use approx::assert_abs_diff_eq;

use crate::*;

#[derive(Clone)]
struct Test {
    infile: String,
    fort15: String,
    fort40: String,
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
            fort40: String::from(
                start.join(dir).join("fort.40").to_str().unwrap(),
            ),
            want: load_fc34(start.join(dir).join("f4qcm")),
            eps,
        }
    }
}

#[test]
pub(crate) fn test_force4() {
    let tests = [
        Test::new("h2o", 2.2e-6),
        Test::new("h2co", 1.7e-6),
        Test::new("c3h2", 1.8e-6),
        Test::new("c3hf", 3.1e-6),
        Test::new("c3hcn", 3.2e-6),
    ];
    for test in tests {
        let s = Spectro::load(&test.infile);
        let fc2 = load_fc2(test.fort15, s.n3n);
        let fc2 = s.rot2nd(fc2);
        let fc2 = FACT2 * fc2;
        let w = s.geom.weights();
        let sqm: Vec<_> = w.iter().map(|w| 1.0 / w.sqrt()).collect();
        let fxm = s.form_sec(fc2, &sqm);
        let (harms, lxm) = symm_eigen_decomp(fxm);
        let freq = to_wavenumbers(harms);
        let lx = s.make_lx(s.n3n, &sqm, &lxm);
        let f4x = load_fc4(test.fort40, s.n3n);
        let mut f4x = s.rot4th(f4x, s.axes);
        let got = force4(s.n3n, &mut f4x, &lx, s.nvib, &freq, s.i4vib);
        let got = Dvec::from(got).abs();
        let want = Dvec::from(test.want).abs();
        // println!("\ndiff = {:.2e}", (got.clone() - want.clone()).max());
        assert_abs_diff_eq!(got, want, epsilon = test.eps);
    }
}

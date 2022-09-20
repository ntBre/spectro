use std::path::Path;

use approx::{abs_diff_eq, assert_abs_diff_eq};

use crate::{consts::FACT2, *};

use super::load_dmat;

#[derive(Clone)]
struct Test {
    infile: String,
    fort15: String,
    fort30: String,
    fort40: String,
    xcnst: Dmat,
    gcnst: Dmat,
    e0: f64,
}

impl Test {
    fn new(dir: &'static str, nvib: usize, e0: f64) -> Self {
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
            fort40: String::from(
                start.join(dir).join("fort.40").to_str().unwrap(),
            ),
            xcnst: load_dmat(start.join(dir).join("xcnst"), nvib, nvib),
            gcnst: load_dmat(start.join(dir).join("gcnst"), nvib, nvib),
            e0,
        }
    }
}

macro_rules! check {
    ($got: expr, $want: expr, $eps: expr, $label: expr, $infile: expr) => {
        if !abs_diff_eq!($got, $want, epsilon = $eps) {
            println!("got\n{:.6}", $got);
            println!("want\n{:.6}", $want);
            println!("diff\n={:.6}", $got.clone() - $want.clone());
            println!(
                "max diff = {:.2e}",
                ($got.clone() - $want.clone()).abs().max()
            );
            assert!(false, "{} differs on {}", $label, $infile);
        }
    };
}

#[test]
fn sym() {
    let tests = [
        Test::new("nh3", 6, 24.716378286389887),
        Test::new("ph3", 6, 20.748849036017717),
        Test::new("bipy", 15, 32.906770783666872),
    ];
    for test in Vec::from(&tests[..]) {
        let s = Spectro::load(&test.infile);
        let fc2 = load_fc2(test.fort15, s.n3n);
        let fc2 = s.rot2nd(fc2);
        let fc2 = FACT2 * fc2;
        let w = s.geom.weights();
        let sqm: Vec<_> = w.iter().map(|w| 1.0 / w.sqrt()).collect();
        let fxm = s.form_sec(fc2, &sqm);
        let (harms, mut lxm) = utils::linalg::symm_eigen_decomp(fxm, true);
        let freq = to_wavenumbers(&harms);
        let mut lx = s.make_lx(&sqm, &lxm);
        s.bdegnl(&freq, &mut lxm, &w, &mut lx);
        let (zmat, wila) = s.zeta(&lxm, &w);
        let f3x = load_fc3(test.fort30, s.n3n);
        let mut f3x = s.rot3rd(f3x);
        let f3qcm = force3(s.n3n, &mut f3x, &lx, s.nvib, &freq);
        let f4x = load_fc4(test.fort40, s.n3n);
        let f4x = s.rot4th(f4x);
        let f4qcm = force4(s.n3n, &f4x, &lx, s.nvib, &freq);
        let Restst {
            coriolis: _,
            fermi1,
            fermi2,
            darling: _,
            states: _,
            modes,
            ifunda: _,
            iovrtn: _,
            icombn: _,
        } = Restst::new(&s, &zmat, &f3qcm, &freq);
        let (xcnst, gcnst, e0) = s.xcals(
            &f4qcm, &freq, &f3qcm, &zmat, &fermi1, &fermi2, &modes, &wila,
        );
        assert_abs_diff_eq!(e0, test.e0, epsilon = 2e-9);
        check!(&xcnst, &test.xcnst, 5e-10, "xcnst", &test.infile);
        check!(&gcnst, &test.gcnst, 3e-9, "gcnst", &test.infile);
    }
}

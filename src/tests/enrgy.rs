use std::path::{Path, PathBuf};

use crate::consts::FACT2;
use crate::utils::linalg::symm_eigen_decomp;
use crate::*;

#[derive(Clone)]
struct Test {
    infile: PathBuf,
    fort15: PathBuf,
    fort30: PathBuf,
    fort40: PathBuf,
    want: Vec<f64>,
}

impl Test {
    fn new(dir: &'static str) -> Self {
        let start = Path::new("testfiles");
        Self {
            infile: start.join(dir).join("spectro.in"),
            fort15: start.join(dir).join("fort.15"),
            fort30: start.join(dir).join("fort.30"),
            fort40: start.join(dir).join("fort.40"),
            want: load_vec(start.join(dir).join("enrgy")),
        }
    }
}

#[test]
fn full() {
    let tests = [
        //
        Test::new("h2o"),
        Test::new("bipy"),
    ];
    for test in tests {
        let s = Spectro::load(test.infile);
        let fc2 = load_fc2(test.fort15, s.n3n);
        let fc2 = s.rot2nd(fc2);
        let fc2 = FACT2 * fc2;
        let w = s.geom.weights();
        let sqm: Vec<_> = w.iter().map(|w| 1.0 / w.sqrt()).collect();
        let fxm = s.form_sec(fc2, &sqm);
        let (harms, mut lxm) = symm_eigen_decomp(fxm, true);
        let freq = to_wavenumbers(&harms);
        let mut lx = s.make_lx(&sqm, &lxm);
        if s.rotor.is_sym_top() {
            s.bdegnl(&freq, &mut lxm, &w, &mut lx);
        }
        let f3x = load_fc3(test.fort30, s.n3n);
        let mut f3x = s.rot3rd(f3x);
        let f3qcm = force3(s.n3n, &mut f3x, &lx, s.nvib, &freq);
        let f4x = load_fc4(test.fort40, s.n3n);
        let f4x = s.rot4th(f4x);
        let f4qcm = force4(s.n3n, &f4x, &lx, s.nvib, &freq);
        let (zmat, wila) = s.zeta(&lxm, &w);
        let restst = Restst::new(&s, &zmat, &f3qcm, &freq);
        let Restst {
            coriolis: _,
            fermi1,
            fermi2,
            darling: _,
            modes,
            states,
            ifunda: _,
            iovrtn: _,
            icombn: _,
        } = &restst;
        let (xcnst, gcnst, e0) = if s.rotor.is_sym_top() {
            let (x, g, e) = s.xcals(
                &f4qcm, &freq, &f3qcm, &zmat, &fermi1, &fermi2, &modes, &wila,
            );
            (x, Some(g), e)
        } else {
            let (x, e) =
                s.xcalc(&f4qcm, &freq, &f3qcm, &zmat, &modes, &fermi1, &fermi2);
            (x, None, e)
        };
        let mut got = vec![0.0; states.len()];
        s.enrgy(&freq, &xcnst, &gcnst, &restst, &f3qcm, e0, &mut got);
        check_vec!(Dvec::from(got), Dvec::from(test.want), 2.1e-5, "enrgy");
    }
}

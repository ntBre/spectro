use crate::*;

struct Test {
    infile: String,
    fort15: String,
    fort30: String,
    want: Restst,
}

impl Test {
    fn new(dir: &'static str, want: Restst) -> Self {
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
            want,
        }
    }
}

#[test]
pub(crate) fn restst() {
    let tests = [
        Test::new(
            "h2o",
            Restst {
                coriolis: vec![],
                fermi1: vec![],
                fermi2: vec![],
                darling: vec![Darling::new(1, 0)],
                i1sts: vec![
                    vec![0, 0, 0],
                    vec![1, 0, 0],
                    vec![0, 1, 0],
                    vec![0, 0, 1],
                    vec![2, 0, 0],
                    vec![0, 2, 0],
                    vec![0, 0, 2],
                    vec![1, 1, 0],
                    vec![1, 0, 1],
                    vec![0, 1, 1],
                ],
                i1mode: vec![0, 1, 2],
            },
        ),
        Test::new(
            "h2co",
            Restst {
                coriolis: vec![Coriolis::new(5, 4, 0)],
                fermi1: vec![Fermi1::new(3, 1)],
                fermi2: vec![Fermi2::new(4, 2, 0)],
                darling: vec![Darling::new(1, 0), Darling::new(5, 4)],
                i1sts: vec![
                    vec![0, 0, 0, 0, 0, 0],
                    vec![1, 0, 0, 0, 0, 0],
                    vec![0, 1, 0, 0, 0, 0],
                    vec![0, 0, 1, 0, 0, 0],
                    vec![0, 0, 0, 1, 0, 0],
                    vec![0, 0, 0, 0, 1, 0],
                    vec![0, 0, 0, 0, 0, 1],
                    vec![2, 0, 0, 0, 0, 0],
                    vec![0, 2, 0, 0, 0, 0],
                    vec![0, 0, 2, 0, 0, 0],
                    vec![0, 0, 0, 2, 0, 0],
                    vec![0, 0, 0, 0, 2, 0],
                    vec![0, 0, 0, 0, 0, 2],
                    vec![1, 1, 0, 0, 0, 0],
                    vec![1, 0, 1, 0, 0, 0],
                    vec![1, 0, 0, 1, 0, 0],
                    vec![1, 0, 0, 0, 1, 0],
                    vec![1, 0, 0, 0, 0, 1],
                    vec![0, 1, 1, 0, 0, 0],
                    vec![0, 1, 0, 1, 0, 0],
                    vec![0, 1, 0, 0, 1, 0],
                    vec![0, 1, 0, 0, 0, 1],
                    vec![0, 0, 1, 1, 0, 0],
                    vec![0, 0, 1, 0, 1, 0],
                    vec![0, 0, 1, 0, 0, 1],
                    vec![0, 0, 0, 1, 1, 0],
                    vec![0, 0, 0, 1, 0, 1],
                    vec![0, 0, 0, 0, 1, 1],
                ],
                i1mode: vec![0, 1, 2, 3, 4, 5],
            },
        ),
        Test::new(
            "c3h2",
            Restst {
                coriolis: vec![
                    Coriolis::new(5, 4, 0),
                    Coriolis::new(6, 5, 0),
                    Coriolis::new(7, 4, 2),
                    Coriolis::new(7, 5, 1),
                    Coriolis::new(7, 6, 2),
                    Coriolis::new(8, 6, 1),
                    Coriolis::new(8, 7, 0),
                ],
                fermi1: vec![
                    Fermi1::new(2, 0),
                    Fermi1::new(6, 2),
                    Fermi1::new(7, 2),
                    Fermi1::new(8, 2),
                ],
                fermi2: vec![],
                darling: vec![
                    Darling::new(1, 0),
                    Darling::new(5, 4),
                    Darling::new(6, 5),
                    Darling::new(7, 5),
                    Darling::new(7, 6),
                    Darling::new(8, 6),
                    Darling::new(8, 7),
                ],
                i1sts: vec![
                    vec![0, 0, 0, 0, 0, 0, 0, 0, 0],
                    vec![1, 0, 0, 0, 0, 0, 0, 0, 0],
                    vec![0, 1, 0, 0, 0, 0, 0, 0, 0],
                    vec![0, 0, 1, 0, 0, 0, 0, 0, 0],
                    vec![0, 0, 0, 1, 0, 0, 0, 0, 0],
                    vec![0, 0, 0, 0, 1, 0, 0, 0, 0],
                    vec![0, 0, 0, 0, 0, 1, 0, 0, 0],
                    vec![0, 0, 0, 0, 0, 0, 1, 0, 0],
                    vec![0, 0, 0, 0, 0, 0, 0, 1, 0],
                    vec![0, 0, 0, 0, 0, 0, 0, 0, 1],
                    vec![2, 0, 0, 0, 0, 0, 0, 0, 0],
                    vec![0, 2, 0, 0, 0, 0, 0, 0, 0],
                    vec![0, 0, 2, 0, 0, 0, 0, 0, 0],
                    vec![0, 0, 0, 2, 0, 0, 0, 0, 0],
                    vec![0, 0, 0, 0, 2, 0, 0, 0, 0],
                    vec![0, 0, 0, 0, 0, 2, 0, 0, 0],
                    vec![0, 0, 0, 0, 0, 0, 2, 0, 0],
                    vec![0, 0, 0, 0, 0, 0, 0, 2, 0],
                    vec![0, 0, 0, 0, 0, 0, 0, 0, 2],
                    vec![1, 1, 0, 0, 0, 0, 0, 0, 0],
                    vec![1, 0, 1, 0, 0, 0, 0, 0, 0],
                    vec![1, 0, 0, 1, 0, 0, 0, 0, 0],
                    vec![1, 0, 0, 0, 1, 0, 0, 0, 0],
                    vec![1, 0, 0, 0, 0, 1, 0, 0, 0],
                    vec![1, 0, 0, 0, 0, 0, 1, 0, 0],
                    vec![1, 0, 0, 0, 0, 0, 0, 1, 0],
                    vec![1, 0, 0, 0, 0, 0, 0, 0, 1],
                    vec![0, 1, 1, 0, 0, 0, 0, 0, 0],
                    vec![0, 1, 0, 1, 0, 0, 0, 0, 0],
                    vec![0, 1, 0, 0, 1, 0, 0, 0, 0],
                    vec![0, 1, 0, 0, 0, 1, 0, 0, 0],
                    vec![0, 1, 0, 0, 0, 0, 1, 0, 0],
                    vec![0, 1, 0, 0, 0, 0, 0, 1, 0],
                    vec![0, 1, 0, 0, 0, 0, 0, 0, 1],
                    vec![0, 0, 1, 1, 0, 0, 0, 0, 0],
                    vec![0, 0, 1, 0, 1, 0, 0, 0, 0],
                    vec![0, 0, 1, 0, 0, 1, 0, 0, 0],
                    vec![0, 0, 1, 0, 0, 0, 1, 0, 0],
                    vec![0, 0, 1, 0, 0, 0, 0, 1, 0],
                    vec![0, 0, 1, 0, 0, 0, 0, 0, 1],
                    vec![0, 0, 0, 1, 1, 0, 0, 0, 0],
                    vec![0, 0, 0, 1, 0, 1, 0, 0, 0],
                    vec![0, 0, 0, 1, 0, 0, 1, 0, 0],
                    vec![0, 0, 0, 1, 0, 0, 0, 1, 0],
                    vec![0, 0, 0, 1, 0, 0, 0, 0, 1],
                    vec![0, 0, 0, 0, 1, 1, 0, 0, 0],
                    vec![0, 0, 0, 0, 1, 0, 1, 0, 0],
                    vec![0, 0, 0, 0, 1, 0, 0, 1, 0],
                    vec![0, 0, 0, 0, 1, 0, 0, 0, 1],
                    vec![0, 0, 0, 0, 0, 1, 1, 0, 0],
                    vec![0, 0, 0, 0, 0, 1, 0, 1, 0],
                    vec![0, 0, 0, 0, 0, 1, 0, 0, 1],
                    vec![0, 0, 0, 0, 0, 0, 1, 1, 0],
                    vec![0, 0, 0, 0, 0, 0, 1, 0, 1],
                    vec![0, 0, 0, 0, 0, 0, 0, 1, 1],
                ],
                i1mode: vec![0, 1, 2, 3, 4, 5, 6, 7, 8],
            },
        ),
        Test::new(
            "c3hf",
            Restst {
                coriolis: vec![
                    Coriolis::new(5, 4, 0),
                    Coriolis::new(5, 4, 1),
                    Coriolis::new(8, 7, 0),
                ],
                fermi1: vec![
                    Fermi1::new(4, 1),
                    Fermi1::new(5, 1),
                    Fermi1::new(6, 2),
                    Fermi1::new(7, 6),
                    Fermi1::new(8, 4),
                    Fermi1::new(8, 6),
                ],
                fermi2: vec![
                    Fermi2::new(2, 1, 0),
                    Fermi2::new(6, 3, 1),
                    Fermi2::new(6, 4, 1),
                    Fermi2::new(7, 2, 1),
                    Fermi2::new(7, 3, 1),
                    Fermi2::new(7, 4, 2),
                    Fermi2::new(7, 6, 2),
                    Fermi2::new(8, 7, 5),
                ],
                darling: vec![
                    Darling::new(5, 4),
                    Darling::new(6, 5),
                    Darling::new(8, 7),
                ],
                i1sts: vec![
                    vec![0, 0, 0, 0, 0, 0, 0, 0, 0],
                    vec![1, 0, 0, 0, 0, 0, 0, 0, 0],
                    vec![0, 1, 0, 0, 0, 0, 0, 0, 0],
                    vec![0, 0, 1, 0, 0, 0, 0, 0, 0],
                    vec![0, 0, 0, 1, 0, 0, 0, 0, 0],
                    vec![0, 0, 0, 0, 1, 0, 0, 0, 0],
                    vec![0, 0, 0, 0, 0, 1, 0, 0, 0],
                    vec![0, 0, 0, 0, 0, 0, 1, 0, 0],
                    vec![0, 0, 0, 0, 0, 0, 0, 1, 0],
                    vec![0, 0, 0, 0, 0, 0, 0, 0, 1],
                    vec![2, 0, 0, 0, 0, 0, 0, 0, 0],
                    vec![0, 2, 0, 0, 0, 0, 0, 0, 0],
                    vec![0, 0, 2, 0, 0, 0, 0, 0, 0],
                    vec![0, 0, 0, 2, 0, 0, 0, 0, 0],
                    vec![0, 0, 0, 0, 2, 0, 0, 0, 0],
                    vec![0, 0, 0, 0, 0, 2, 0, 0, 0],
                    vec![0, 0, 0, 0, 0, 0, 2, 0, 0],
                    vec![0, 0, 0, 0, 0, 0, 0, 2, 0],
                    vec![0, 0, 0, 0, 0, 0, 0, 0, 2],
                    vec![1, 1, 0, 0, 0, 0, 0, 0, 0],
                    vec![1, 0, 1, 0, 0, 0, 0, 0, 0],
                    vec![1, 0, 0, 1, 0, 0, 0, 0, 0],
                    vec![1, 0, 0, 0, 1, 0, 0, 0, 0],
                    vec![1, 0, 0, 0, 0, 1, 0, 0, 0],
                    vec![1, 0, 0, 0, 0, 0, 1, 0, 0],
                    vec![1, 0, 0, 0, 0, 0, 0, 1, 0],
                    vec![1, 0, 0, 0, 0, 0, 0, 0, 1],
                    vec![0, 1, 1, 0, 0, 0, 0, 0, 0],
                    vec![0, 1, 0, 1, 0, 0, 0, 0, 0],
                    vec![0, 1, 0, 0, 1, 0, 0, 0, 0],
                    vec![0, 1, 0, 0, 0, 1, 0, 0, 0],
                    vec![0, 1, 0, 0, 0, 0, 1, 0, 0],
                    vec![0, 1, 0, 0, 0, 0, 0, 1, 0],
                    vec![0, 1, 0, 0, 0, 0, 0, 0, 1],
                    vec![0, 0, 1, 1, 0, 0, 0, 0, 0],
                    vec![0, 0, 1, 0, 1, 0, 0, 0, 0],
                    vec![0, 0, 1, 0, 0, 1, 0, 0, 0],
                    vec![0, 0, 1, 0, 0, 0, 1, 0, 0],
                    vec![0, 0, 1, 0, 0, 0, 0, 1, 0],
                    vec![0, 0, 1, 0, 0, 0, 0, 0, 1],
                    vec![0, 0, 0, 1, 1, 0, 0, 0, 0],
                    vec![0, 0, 0, 1, 0, 1, 0, 0, 0],
                    vec![0, 0, 0, 1, 0, 0, 1, 0, 0],
                    vec![0, 0, 0, 1, 0, 0, 0, 1, 0],
                    vec![0, 0, 0, 1, 0, 0, 0, 0, 1],
                    vec![0, 0, 0, 0, 1, 1, 0, 0, 0],
                    vec![0, 0, 0, 0, 1, 0, 1, 0, 0],
                    vec![0, 0, 0, 0, 1, 0, 0, 1, 0],
                    vec![0, 0, 0, 0, 1, 0, 0, 0, 1],
                    vec![0, 0, 0, 0, 0, 1, 1, 0, 0],
                    vec![0, 0, 0, 0, 0, 1, 0, 1, 0],
                    vec![0, 0, 0, 0, 0, 1, 0, 0, 1],
                    vec![0, 0, 0, 0, 0, 0, 1, 1, 0],
                    vec![0, 0, 0, 0, 0, 0, 1, 0, 1],
                    vec![0, 0, 0, 0, 0, 0, 0, 1, 1],
                ],
                i1mode: vec![0, 1, 2, 3, 4, 5, 6, 7, 8],
            },
        ),
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
        let (zmat, _wila) = s.zeta(&lxm, &w);
        let f3x = load_fc3(test.fort30, s.n3n);
        let mut f3x = s.rot3rd(f3x, s.axes);
        let f3qcm = force3(s.n3n, &mut f3x, &lx, s.nvib, &freq);
        let got = s.restst(&zmat, &f3qcm, &freq);
        assert_eq!(got.coriolis, test.want.coriolis);
        assert_eq!(got.fermi1, test.want.fermi1);
        assert_eq!(got.fermi2, test.want.fermi2);
        assert_eq!(got.darling, test.want.darling);
        assert_eq!(got.i1sts, test.want.i1sts);
        assert_eq!(got.i1mode, test.want.i1mode);
        assert_eq!(got, test.want);
    }
}

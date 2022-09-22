//! tests for building up run step-by-step from the very beginning. To add a new
//! test case:
//!
//! 1. test `Spectro::load` to make sure the right geometry and principal axes
//! are being loaded since those will affect everything else.
//!
//! 2. test `load_fc2` to make sure the correct harmonic force constants are
//! loaded from fort.15
//!
//! 3. test `rot_2nd` to make sure the loaded harmonic force constants are
//! rotated to the new axes correctly
//!
//! 4. test `form_sec` to make sure these rotated force constants are converted
//! to the correct secular equations

use crate::{
    check_mat, consts::FACT2, tests::load_dmat, utils::load_fc2, Dmat, Mat3,
    Spectro,
};
use approx::assert_abs_diff_eq;
use nalgebra::matrix;
use std::{
    path::{Path, PathBuf},
    str::FromStr,
};
use symm::Molecule;

#[test]
fn test_load() {
    #[derive(Clone)]
    struct Test {
        infile: PathBuf,
        want: Want,
    }

    #[derive(Clone)]
    struct Want {
        axes: Mat3,
        geom: Molecule,
    }

    impl Test {
        fn new(dir: &'static str, axes: Mat3, geom: &str) -> Self {
            let start = Path::new("testfiles");
            Self {
                infile: start.join(dir).join("spectro.in"),
                want: Want {
                    axes,
                    geom: Molecule::from_str(geom).unwrap(),
                },
            }
        }
    }

    let tests = [
        //
        Test::new(
            "h2o_sic",
            matrix![
            0.00000000, 0.00000000, 1.00000000;
            1.00000000, 0.00000000, 0.00000000;
            0.00000000, 1.00000000, 0.00000000;
                ],
            "H     -0.7574256      0.5217723      0.0000000
             O      0.0000000     -0.0657528      0.0000000
             H      0.7574256      0.5217723      0.0000000",
        ),
    ];

    for test in tests {
        let got = Spectro::load(&test.infile);
        check_mat!(&got.axes, &test.want.axes, 1e-8, &test.infile.display());
        assert_abs_diff_eq!(got.geom, test.want.geom, epsilon = 1e-6);
    }
}

#[test]
fn test_load_fc2() {
    #[derive(Clone)]
    struct Test {
        infile: PathBuf,
        fort15: PathBuf,
        want: Dmat,
    }

    impl Test {
        fn new(dir: &'static str, n3n: usize) -> Self {
            let start = Path::new("testfiles");
            Self {
                infile: start.join(dir).join("spectro.in"),
                fort15: start.join(dir).join("fort.15"),
                want: load_dmat(start.join(dir).join("step_fc2"), n3n, n3n),
            }
        }
    }

    let tests = [
        //
        Test::new("h2o_sic", 9),
    ];

    for test in tests {
        let s = Spectro::load(&test.infile);
        let got = load_fc2(test.fort15, s.n3n);
        check_mat!(&got, &test.want, 1e-14, &test.infile.display());
    }
}

#[test]
fn test_rot2nd() {
    #[derive(Clone)]
    struct Test {
        infile: PathBuf,
        fort15: PathBuf,
        want: Dmat,
    }

    impl Test {
        fn new(dir: &'static str, n3n: usize) -> Self {
            let start = Path::new("testfiles");
            Self {
                infile: start.join(dir).join("spectro.in"),
                fort15: start.join(dir).join("fort.15"),
                want: load_dmat(start.join(dir).join("step_rot2nd"), n3n, n3n),
            }
        }
    }

    let tests = [
        //
        Test::new("h2o_sic", 9),
    ];

    for test in tests {
        let s = Spectro::load(&test.infile);
        let fc2 = load_fc2(test.fort15, s.n3n);
        let fc2 = s.rot2nd(fc2);
        let got = FACT2 * fc2;
        check_mat!(&got, &test.want, 4e-13, &test.infile.display());
    }
}

#[test]
fn fxm() {
    #[derive(Clone)]
    struct Test {
        infile: PathBuf,
        fort15: PathBuf,
        want: Dmat,
    }

    impl Test {
        fn new(dir: &'static str, n3n: usize) -> Self {
            let start = Path::new("testfiles");
            Self {
                infile: start.join(dir).join("spectro.in"),
                fort15: start.join(dir).join("fort.15"),
                want: load_dmat(start.join(dir).join("step_fxm"), n3n, n3n),
            }
        }
    }

    let tests = [
        //
        Test::new("h2o_sic", 9),
        Test::new("c2h-", 9),
    ];

    for test in tests {
        let s = Spectro::load(&test.infile);
        let fc2 = load_fc2(test.fort15, s.n3n);
        let fc2 = s.rot2nd(fc2);
        let fc2 = FACT2 * fc2;
        let w = s.geom.weights();
        let sqm: Vec<_> = w.iter().map(|w| 1.0 / w.sqrt()).collect();
        let got = s.form_sec(fc2, &sqm);
        let mut want = test.want.clone();
        want.fill_upper_triangle_with_lower_triangle();
        check_mat!(&got, &want, 5e-13, &test.infile.display());
    }
}

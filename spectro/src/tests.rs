//! uncategorized tests and utilities shared by the other modules

use crate::consts::FACT2;
use crate::resonance::Restst;
use crate::utils::linalg::symm_eigen_decomp;
use crate::*;
use approx::{abs_diff_eq, abs_diff_ne, assert_abs_diff_eq};
use na::allocator::Allocator;
use na::{dmatrix, DefaultAllocator, Matrix, Storage};
use nalgebra as na;
use std::fs::read_to_string;
use std::io::{BufRead, BufReader};
use std::ops::Sub;
use std::path::Path;

mod alphaa;
mod alphas;
mod bench;
mod enrgy;
mod force3;
mod force4;
mod load;
mod lxm;
mod quartic;
mod restst;
mod rot2nd;
mod run;
mod secular;
mod sextic;
mod steps;
mod xcalc;
mod xcals;
mod zeta;

pub(crate) fn load_dmat<P: AsRef<Path> + Debug + Clone>(
    filename: P,
    rows: usize,
    cols: usize,
) -> Dmat {
    let data = read_to_string(filename.clone())
        .expect(&format!("failed to read {:?}", filename));
    Dmat::from_iterator(
        cols,
        rows,
        data.split_whitespace().map(|s| s.parse().unwrap()),
    )
    .transpose()
}

/// load a symmetric, square, lower triangular matrix of `size` from `filename`
fn load_lower_triangle(filename: &str, size: usize) -> Dmat {
    let f = std::fs::File::open(filename).unwrap();
    let lines = BufReader::new(f).lines();
    let mut ret = Dmat::zeros(size, size);
    for (i, line) in lines.flatten().enumerate() {
        let sp = line.split_whitespace().map(|s| s.parse::<f64>().unwrap());
        for (j, v) in sp.enumerate() {
            ret[(i, j)] = v;
            ret[(j, i)] = v;
        }
    }
    ret
}

pub(crate) fn check_vec(got: Dvec, want: Dvec, eps: f64, infile: &str) {
    assert_eq!(got.len(), want.len());
    if !abs_diff_eq!(got, want, epsilon = eps) {
        assert_eq!(got.len(), want.len());
        println!(
            "\n{:>5}{:>20.12}{:>20.12}{:>20.12}",
            "Iter", "Got", "Want", "Diff",
        );
        for i in 0..got.len() {
            if (got[i] - want[i]).abs() > eps {
                println!(
                    "{:5}{:20.12}{:20.12}{:20.12}",
                    i,
                    got[i],
                    want[i],
                    got[i] - want[i]
                );
            }
        }
        assert!(
            false,
            "differs by {:.2e} on {}",
            (got.clone() - want.clone()).abs().max(),
            infile
        );
    }
}

#[macro_export]
macro_rules! check_vec {
    ($got: expr, $want: expr, $eps: expr, $infile: expr) => {
        $crate::tests::check_vec(
            $got,
            $want,
            $eps,
            &format!("'{}', {}:{}:{}", $infile, file!(), line!(), column!()),
        )
    };
}

// These sure look similar, but I couldn't figure out the Trait bounds to make
// it generic
fn check_tens(
    got: &Tensor3,
    want: &Tensor3,
    eps: f64,
    label: &str,
    infile: &str,
) {
    if abs_diff_ne!(got, want, epsilon = eps) {
        println!("got\n{:.8}", got);
        println!("want\n{:.8}", want);
        println!(
            "max diff = {:.2e}",
            (got.clone() - want.clone()).abs().max()
        );
        assert!(false, "{} failed on {}", label, infile);
    }
}

#[macro_export]
macro_rules! check_tens {
    ($got: expr, $want: expr, $eps: expr, $label: expr, $infile: expr) => {
        $crate::tests::check_tens(
            $got,
            $want,
            $eps,
            $label,
            &format!("'{}', {}:{}:{}", $infile, file!(), line!(), column!()),
        )
    };
}

pub(crate) fn check_mat<
    'a,
    R: nalgebra::Dim,
    C: nalgebra::Dim,
    S: nalgebra::RawStorage<f64, R, C>,
>(
    got: &'a Matrix<f64, R, C, S>,
    want: &'a Matrix<f64, R, C, S>,
    eps: f64,
    infile: &str,
) where
    &'a Matrix<f64, R, C, S>: Sub<Output = Matrix<f64, R, C, S>>,
    S: Storage<f64, R, C>,
    DefaultAllocator: Allocator<f64, R, C>,
{
    if abs_diff_ne!(got, want, epsilon = eps) {
        println!("got\n{:.8}", got);
        println!("want\n{:.8}", want);
        let diff = got - want;
        println!("diff\n{:.8}", diff);
        println!("max diff = {:.2e}", diff.abs().max());
        panic!("failed on {}", infile);
    }
}

#[macro_export]
macro_rules! check_mat {
    ($got: expr, $want: expr, $eps: expr, $infile: expr) => {
        $crate::tests::check_mat(
            $got,
            $want,
            $eps,
            &format!("'{}', {}:{}:{}", $infile, file!(), line!(), column!()),
        )
    };
}

#[test]
fn test_load_fc2() {
    let got = load_fc2("testfiles/fort.15", 9);
    let want = dmatrix![
    3.2416100000000003e-05, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
    0.0, 0.37452424350000002, 0.23501356370000001, 0.0,
    -0.34333094920000001,
    -0.20370165770000001, 0.0, -0.031192384199999999, -0.031313796200000001;
    0.0, 0.23501356370000001, 0.21818269339999999, -2.7999999999999998e-09,
    -0.26632741869999998, -0.2298216998, 0.0, 0.031313796200000001,
    0.011639992;
    0.0, 0.0, -2.7999999999999998e-09, 6.36953e-05, 0.0,
    2.7999999999999998e-09, -3.0777900000000002e-05, 0.0, 0.0;
    0.0, -0.34333094920000001, -0.26632741869999998, 0.0,
    0.68666005870000002,
    0.0, 0.0, -0.34333094920000001, 0.26632741869999998;
    0.0, -0.20370165770000001, -0.2298216998, 2.7999999999999998e-09, 0.0,
    0.45964140840000001, 0.0, 0.20370165770000001, -0.2298216998;
    0.0, 0.0, 0.0, -3.0777900000000002e-05, 0.0, 0.0,
    3.2416100000000003e-05,
    0.0, 2.7999999999999998e-09;
    0.0, -0.031192384199999999, 0.031313796200000001, 0.0,
    -0.34333094920000001, 0.20370165770000001, 0.0, 0.37452424350000002,
    -0.23501356649999999;
    0.0, -0.031313796200000001, 0.011639992, 0.0, 0.26632741869999998,
    -0.2298216998, 2.7999999999999998e-09, -0.23501356649999999,
    0.21818269339999999;
        ];
    assert_eq!(got, want);
}

#[test]
fn test_funds_and_e0() {
    #[derive(Clone)]
    struct Test {
        infile: &'static str,
        fort15: &'static str,
        fort30: &'static str,
        fort40: &'static str,
        want_e0: f64,
        want_fund: Vec<f64>,
        e_eps: f64,
    }
    let tests = [
        Test {
            infile: "testfiles/h2o/spectro.in",
            fort15: "testfiles/h2o/fort.15",
            fort30: "testfiles/h2o/fort.30",
            fort40: "testfiles/h2o/fort.40",
            want_e0: 20.057563725859055,
            want_fund: vec![3753.166, 3656.537, 1598.516],
            e_eps: 6e-8,
        },
        Test {
            infile: "testfiles/h2co/spectro.in",
            fort15: "testfiles/h2co/fort.15",
            fort30: "testfiles/h2co/fort.30",
            fort40: "testfiles/h2co/fort.40",
            want_e0: 11.49172492996696,
            want_fund: vec![
                2842.9498684325331,
                2780.0927479723691,
                1747.8239712488792,
                1499.4165366400482,
                1246.8067957023538,
                1166.9312314784524,
            ],
            e_eps: 6e-8,
        },
        Test {
            infile: "testfiles/c3h2/spectro.in",
            fort15: "testfiles/c3h2/fort.15",
            fort30: "testfiles/c3h2/fort.30",
            fort40: "testfiles/c3h2/fort.40",
            want_e0: 4.2142433303609623,
            want_fund: vec![
                3140.1433372410634,
                3113.2905071971295,
                1589.145000387535,
                1273.1343017671454,
                1059.5183326828769,
                967.49835061508804,
                887.12367115864799,
                846.15735546423639,
                769.62107643057936,
            ],
            e_eps: 1e-8,
        },
    ];
    for test in Vec::from(&tests[..]) {
        let s = Spectro::load(test.infile);
        let fc2 = load_fc2(test.fort15, s.n3n);
        let fc2 = s.rot2nd(fc2);
        let fc2 = FACT2 * fc2;
        let w = s.geom.weights();
        let sqm: Vec<_> = w.iter().map(|w| 1.0 / w.sqrt()).collect();
        let fxm = s.form_sec(fc2, &sqm);
        let (harms, lxm) = symm_eigen_decomp(fxm, true);
        let freq = to_wavenumbers(&harms);
        let lx = s.make_lx(&sqm, &lxm);
        let (zmat, _wila) = s.zeta(&lxm, &w);
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
            modes,
            states: _,
            ifunda: _,
            iovrtn: _,
            icombn: _,
        } = Restst::new(&s, &zmat, &f3qcm, &freq);
        let (xcnst, e0) =
            s.xcalc(&f4qcm, &freq, &f3qcm, &zmat, &modes, &fermi1, &fermi2);
        assert_abs_diff_eq!(e0, test.want_e0, epsilon = test.e_eps);

        let got = make_funds(&freq, s.nvib, &xcnst);
        assert_abs_diff_eq!(
            Dvec::from(got),
            Dvec::from(test.want_fund),
            epsilon = 1e-3
        );
    }
}
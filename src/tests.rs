use std::str::FromStr;

use approx::assert_abs_diff_eq;
use symm::Molecule;

use crate::{dummy::Dummy, dummy::DummyVal, load_fc2, Spectro, FACT2};

use na::dmatrix;
use nalgebra as na;

#[test]
fn load() {
    let got = Spectro::load("testfiles/spectro.in");
    let want = Spectro {
        header: vec![
            1, 1, 5, 2, 0, 0, 2, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
            0, 0, 0, 0, 0, 0, 0, 0,
        ],
        geom: Molecule::from_str(
            "
 C     -0.00000000        1.69593625        0.00000000
 C      1.27805254        4.09067473        0.00000000
 C     -1.27805254        4.09067473        0.00000000
 H      2.94342058        5.18191475       -0.00000000
 H     -2.94342058        5.18191475        0.00000000",
        )
        .unwrap(),
        weights: vec![
            (1, 12.0),
            (2, 12.0),
            (3, 12.0),
            (4, 1.007825),
            (5, 1.007825),
        ],
        curvils: vec![
            vec![2, 3],
            vec![1, 2],
            vec![1, 3],
            vec![2, 4],
            vec![3, 5],
            vec![2, 4, 1],
            vec![3, 5, 1],
            vec![4, 2, 1, 3],
            vec![5, 3, 1, 2],
        ],
        degmodes: vec![],
        dummies: vec![],
    };
    assert_eq!(got, want);
}

#[test]
fn load_dummy() {
    let got = Spectro::load("testfiles/dummy.in");
    let want = Spectro {
        header: vec![
            1, 1, 3, 0, 2, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
            0, 0, 0, 0, 0, 0, 0, 0,
        ],
        geom: Molecule::from_str(
            "
 He      0.0000000000        0.0000000000       -3.3410028680
 H      0.0000000000        0.0000000000       -1.0030394677
 H      0.0000000000        0.0000000000        1.0030394709
 He      0.0000000000        0.0000000000        3.3410028648
",
        )
        .unwrap(),
        weights: vec![
            (1, 4.00260325413),
            (2, 1.007825),
            (3, 1.007825),
            (4, 4.00260325413),
            (5, 0.00),
            (6, 0.00),
            (7, 0.00),
            (8, 0.00),
        ],
        curvils: vec![
            vec![1, 2],
            vec![2, 3],
            vec![3, 4],
            vec![2, 1, 3],
            vec![3, 2, 4],
        ],
        degmodes: vec![vec![3, 2, 0], vec![1, 2, 3], vec![4, 6], vec![5, 7]],
        dummies: vec![
            Dummy {
                x: DummyVal::Value(1.1111111111),
                y: DummyVal::Atom(3),
                z: DummyVal::Atom(1),
            },
            Dummy {
                x: DummyVal::Atom(3),
                y: DummyVal::Value(1.1111111111),
                z: DummyVal::Atom(1),
            },
            Dummy {
                x: DummyVal::Value(1.1111111111),
                y: DummyVal::Atom(3),
                z: DummyVal::Atom(2),
            },
            Dummy {
                y: DummyVal::Value(1.1111111111),
                x: DummyVal::Atom(3),
                z: DummyVal::Atom(2),
            },
        ],
    };
    assert_eq!(got.curvils, want.curvils);
    assert_eq!(got, want);
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
fn test_rot2nd() {
    let mut spectro = Spectro::load("testfiles/h2o.in");
    spectro.geom.to_angstrom();
    spectro.geom.normalize();
    let axes = spectro.geom.reorder();
    let fc2 = load_fc2("testfiles/fort.15", 9);
    let got = spectro.rot2nd(fc2, axes);
    let want = dmatrix![
    0.37452424350000002, 0.23501356370000001, 0.0, -0.34333094920000001,
    -0.20370165770000001, 0.0, -0.031192384199999999, -0.031313796200000001, 0.0;
    0.23501356370000001, 0.21818269339999999, 0.0, -0.26632741869999998,
    -0.2298216998, -2.7999999999999998e-09, 0.031313796200000001, 0.011639992, 0.0;
     0.0, 0.0, 3.2416100000000003e-05, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
    -0.34333094920000001, -0.26632741869999998, 0.0, 0.68666005870000002,
    0.0, 0.0, -0.34333094920000001, 0.26632741869999998, 0.0;
    -0.20370165770000001, -0.2298216998, 0.0, 0.0, 0.45964140840000001,
    2.7999999999999998e-09, 0.20370165770000001, -0.2298216998, 0.0;
    0.0, -2.7999999999999998e-09, 0.0, 0.0, 2.7999999999999998e-09, 6.36953e-05,
    0.0, 0.0, -3.0777900000000002e-05;
    -0.031192384199999999, 0.031313796200000001, 0.0, -0.34333094920000001,
    0.20370165770000001, 0.0, 0.37452424350000002, -0.23501356649999999, 0.0;
    -0.031313796200000001, 0.011639992, 0.0, 0.26632741869999998,
    -0.2298216998, 0.0, -0.23501356649999999, 0.21818269339999999,
    2.7999999999999998e-09;
     0.0, 0.0, 0.0, 0.0, 0.0, -3.0777900000000002e-05, 0.0,
     2.7999999999999998e-09, 3.2416100000000003e-05;
           ];
    assert_eq!(got, want);
}

#[test]
fn test_sec() {
    let mut spectro = Spectro::load("testfiles/h2o.in");
    spectro.geom.to_angstrom();
    spectro.geom.normalize();
    let axes = spectro.geom.reorder();
    let fc2 = load_fc2("testfiles/fort.15", 9);
    let fc2 = spectro.rot2nd(fc2, axes);
    let fc2 = FACT2 * fc2;
    let n3n = 3 * spectro.natoms();
    let got = spectro.form_sec(fc2, n3n);
    let mut want = dmatrix![
    5.7857638, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
    3.6305606, 3.3705523, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
    0.0000000, 0.0000000, 0.0005008, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
    -1.3313595, -1.0327573, 0.0000000, 0.6683836, 0.0, 0.0, 0.0, 0.0, 0.0;
    -0.7899088, -0.8911964, 0.0000000, 0.0000000, 0.4474074, 0.0, 0.0, 0.0, 0.0;
    0.0000000, -0.0000000, 0.0000000, 0.0000000, 0.0000000, 0.0000620, 0.0, 0.0, 0.0;
    -0.4818694, 0.4837450, 0.0000000, -1.3313595, 0.7899088, 0.0000000, 5.7857638, 0.0, 0.0;
    -0.4837450, 0.1798181, 0.0000000, 1.0327573, -0.8911964, 0.0000000, -3.6305607, 3.3705523, 0.0;
    0.0000000, 0.0000000, 0.0000000, 0.0000000, 0.0000000, -0.0001193, 0.0000000, 0.0000000, 0.0005008;
       ];
    want.fill_upper_triangle_with_lower_triangle();
    assert_abs_diff_eq!(got, want, epsilon = 1e-7);
}

#[test]
fn test_run() {
    let spectro = Spectro::load("testfiles/h2o.in");
    spectro.run();
}

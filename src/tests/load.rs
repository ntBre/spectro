use std::str::FromStr;

use crate::dummy::DummyVal;
use crate::load::process_geom;
use crate::Curvil::*;
use crate::*;

use approx::assert_abs_diff_eq;
use na::matrix;
use nalgebra as na;

use super::check_mat;

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
C      0.00000000 -0.89784221  0.00000000
C      0.67631628  0.36939882  0.00000000
C     -0.67631628  0.36939882  0.00000000
H      1.55759109  0.94685817  0.00000000
H     -1.55759109  0.94685817  0.00000000
",
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
            Bond(2, 3),
            Bond(1, 2),
            Bond(1, 3),
            Bond(2, 4),
            Bond(3, 5),
            Bend(2, 4, 1),
            Bend(3, 5, 1),
            Tors(4, 2, 1, 3),
            Tors(5, 3, 1, 2),
        ],
        degmodes: vec![],
        dummies: vec![],
        rotor: Rotor::AsymmTop,
        n3n: 15,
        i3n3n: 680,
        i4n3n: 3060,
        nvib: 9,
        i2vib: 45,
        i3vib: 165,
        i4vib: 495,
        natom: 5,
        axes: matrix![
        1.0, 0.0, 0.0;
        0.0, 1.0, 0.0;
        0.0, 0.0, 1.0;
        ],
        rotcon: vec![
            1.1424648894465075,
            1.0623773040516062,
            0.5504832830226929,
        ],
        primat: vec![14.75549102256656, 15.867837495713816, 30.62332851828037],
        iatom: 0,
        axis_order: 0,
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
He    -1.76798258  0.00000000  0.00000000
H     -0.53078563  0.00000000  0.00000000
H      0.53078563  0.00000000  0.00000000
He     1.76798258  0.00000000  0.00000000
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
            Bond(1, 2),
            Bond(2, 3),
            Bond(3, 4),
            Bend(2, 1, 3),
            Bend(3, 2, 4),
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
        rotor: Rotor::Linear,
        n3n: 12,
        i3n3n: 364,
        i4n3n: 1365,
        nvib: 7,
        i2vib: 28,
        i3vib: 84,
        i4vib: 210,
        natom: 4,
        axes: matrix![
        0.0, 1.0, 0.0;
        0.0, 0.0, 1.0;
        1.0, 0.0, 0.0;
        ],
        primat: vec![0.0, 19.422657990598942, 19.422657990598942],
        rotcon: vec![
            std::f64::INFINITY,
            0.8679363261189563,
            0.8679363261189563,
        ],
        iatom: 0,
        axis_order: 0,
    };
    assert_eq!(got.axes, want.axes);
    assert_eq!(got, want);
}

struct Test {
    label: String,
    geom: Molecule,
    want_geom: Molecule,
    want_axes: Mat3,
}

impl Test {
    fn new(label: &str, geom: &str, want_geom: &str, want_axes: Mat3) -> Self {
        Self {
            label: String::from(label),
            geom: Molecule::from_str(geom).unwrap(),
            want_geom: Molecule::from_str(want_geom).unwrap(),
            want_axes,
        }
    }
}

/// test the geometry and axes produced by process_geom. output from
/// spectro2.out with input from spectro2.in
#[test]
fn geom_axes() {
    let tests = [
        Test::new(
            "h2o_sic",
            "
         H      0.0000000000      1.4313273344      0.9860352735
         O      0.0000000000      0.0000000000     -0.1242266321
         H      0.0000000000     -1.4313273344      0.9860352735
        ",
            "
        H     -0.7574256      0.5217723      0.0000000
        O      0.0000000     -0.0657528      0.0000000
        H      0.7574256      0.5217723      0.0000000
        ",
            matrix![
            0.00000000,0.00000000,1.00000000;
            -1.00000000,0.00000000,0.00000000;
            0.00000000,1.00000000,0.00000000;
                            ],
        ),
        Test::new(
            "c3hcn",
            "
 H      0.0000000000      3.0019203461      3.8440336302
 C      0.0000000000      1.2175350291      2.8648781120
 C      0.0000000000     -1.4653360811      2.9704535522
 C      0.0000000000     -0.0243525793      0.6850185070
 C      0.0000000000     -0.0005362006     -1.9780266119
 N      0.0000000000      0.0178435988     -4.1716979516
",
            "
H      2.0345490     -1.5884146      0.0000000
C      1.5163524     -0.6441862      0.0000000
C      1.5721454      0.7755306      0.0000000
C      0.3627860      0.0129312      0.0000000
C     -1.0464358      0.0002536      0.0000000
N     -2.2072758     -0.0095340      0.0000000
",
            matrix![
            0.00000000,0.00000000,1.00000000;
            0.00005289,-1.00000000,0.00000000;
            1.00000000,0.00005289,0.00000000;
                ],
        ),
    ];
    for test in tests {
        let mut s = Spectro {
            geom: test.geom,
            ..Spectro::default()
        };
        process_geom(&mut s);
        check_mat!(&s.axes, &test.want_axes, 1e-8, "geom_axes", &test.label);
        assert_abs_diff_eq!(s.geom, test.want_geom, epsilon = 1e-6);
    }
}

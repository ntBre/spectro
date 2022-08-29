use std::str::FromStr;

use crate::Curvil::*;
use crate::*;

use na::matrix;
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
        rotcon: vec![
            std::f64::INFINITY,
            0.8679363261189563,
            0.8679363261189563,
        ],
        primat: vec![0.0, 19.422657990598942, 19.422657990598942],
    };
    assert_eq!(got.axes, want.axes);
    assert_eq!(got, want);
}

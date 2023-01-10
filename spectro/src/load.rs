use std::{
    collections::HashMap,
    fmt::Debug,
    fs::{read_to_string, File},
    io::{BufRead, BufReader},
    path::Path,
};

use crate::Spectro;
use crate::{
    consts::CONST,
    dummy::{Dummy, DummyVal},
};
use crate::{utils::*, Dmat, Tensor3};
use symm::{Atom, Axis, Molecule, Plane};
use tensor::Tensor4;

impl Spectro {
    pub fn load<P>(filename: P) -> Self
    where
        P: AsRef<Path> + Debug + Clone,
    {
        let mut ret = read(filename);
        process_geom(&mut ret);
        ret
    }
}

impl From<Molecule> for Spectro {
    fn from(geom: Molecule) -> Self {
        let mut ret = Self {
            geom,
            ..Self::default()
        };
        process_geom(&mut ret);
        ret
    }
}

/// perform the geometry manipulations from dist.f on ret.geom and set the
/// corresponding fields in `ret`. assumes the input geometry is in bohr and
/// immediately converts to angstroms
pub(crate) fn process_geom(ret: &mut Spectro) {
    // assumes input geometry in bohr
    ret.geom.to_angstrom();
    let (pr, axes, rotor) = ret.geom.normalize();
    ret.rotor = rotor;
    ret.primat = Vec::from(pr.as_slice());
    ret.rotcon = pr
        .iter()
        .map(|m| if *m > 1e-2 { CONST / m } else { 0.0 })
        .collect();
    ret.axes = axes;

    // center of mass again
    let com = ret.geom.com();
    ret.geom.translate(-com);
    // detect atom not on principal axis and bisected by a mirror plane, in
    // other words, the coordinate on the axis perpendicular to the mirror
    // plane is zero AND the other coordinate not on the axis is non-zero
    //
    // for example, in the ammonia geometry:
    //
    // H     -0.93663636 -0.00000000 -0.31301502
    // N     -0.00000001  0.00000000  0.06758466
    // H      0.46831825 -0.81115089 -0.31301496
    // H      0.46831825  0.81115089 -0.31301496
    //
    // both H1 and N meet the first criterion, both have Y = 0, but N also
    // has X = 0, which breaks the second criterion, meaning we should
    // rotate H1 to the X axis, but it's already there
    if ret.rotor.is_sym_top() && !ret.rotor.is_linear() {
        // NOTE smallest eps for which this works for the current test cases
        const TOL: f64 = 1e-6;
        let pg = ret.geom.point_group_approx(TOL);
        use symm::PointGroup::*;
        fn x3x_helper(ret: &mut Spectro, axis: Axis, plane: Plane) -> usize {
            // axis perpendicular to plane
            let p = plane.perp();
            // axis in plane but not principal axis
            let o = plane ^ axis;

            ret.axis_order = 3;
            ret.geom
                .atoms
                .iter()
                .position(|a| {
                    let v = [a.x, a.y, a.z];
                    v[p as usize].abs() < TOL && v[o as usize].abs() > TOL
                })
                .unwrap()
        }
        let iatl = match pg {
            C1 => todo!(),
            C2 { axis: _ } => todo!(),
            Cs { plane: _ } => todo!(),
            C2v { axis: _, planes: _ } => todo!(),
            C3v { axis, plane } => x3x_helper(ret, axis, plane),
            D2h { axes: _, planes: _ } => todo!(),
            // assume that these are in the right order
            D3h {
                c3,
                c2: _,
                sh: _,
                sv,
            } => x3x_helper(ret, c3, sv),
        };
        let mut egr = nalgebra::Matrix3::zeros();
        let x = ret.geom.atoms[iatl].x;
        let y = ret.geom.atoms[iatl].y;
        egr[(0, 0)] = x / (f64::sqrt(x * x + y * y));
        egr[(1, 0)] = y / (f64::sqrt(x * x + y * y));
        egr[(0, 1)] = -y / (f64::sqrt(x * x + y * y));
        egr[(1, 1)] = x / (f64::sqrt(x * x + y * y));
        egr[(2, 2)] = 1.0;

        for i in 0..ret.natoms() {
            let crot1 = ret.geom.atoms[i].x;
            let crot2 = ret.geom.atoms[i].y;
            let crot3 = ret.geom.atoms[i].z;
            ret.geom.atoms[i].x =
                egr[(0, 0)] * crot1 + egr[(1, 0)] * crot2 + egr[(2, 0)] * crot3;
            ret.geom.atoms[i].y =
                egr[(0, 1)] * crot1 + egr[(1, 1)] * crot2 + egr[(2, 1)] * crot3;
            ret.geom.atoms[i].z =
                egr[(0, 2)] * crot1 + egr[(1, 2)] * crot2 + egr[(2, 2)] * crot3;
        }

        let btemp = egr.transpose() * ret.axes.transpose();
        ret.axes = btemp.transpose();
        ret.iatom = iatl;

        // detect point group and store the principal (C₃) axis for later use
        // with iatl. have to do this again after the geometry is rotated
        ret.axis = match ret.geom.point_group_approx(TOL) {
            C1 => todo!(),
            C2 { axis: _ } => todo!(),
            Cs { plane: _ } => todo!(),
            C2v { axis: _, planes: _ } => todo!(),
            C3v { axis, plane: _ } => axis,
            D2h { axes: _, planes: _ } => todo!(),
            D3h {
                c3,
                c2: _,
                sh: _,
                sv: _,
            } => c3,
        };
    }
    // linear molecules should have the unique moi in the Z position. in case x
    // and y got swapped (since their mois are equal), swap them back to keep
    // the original order
    if ret.rotor.is_linear() {
        ret.axes.set_column(0, &nalgebra::vector![1.0, 0.0, 0.0]);
        ret.axes.set_column(1, &nalgebra::vector![0.0, 1.0, 0.0]);
    }
    ret.natom = ret.natoms();
    let n3n = 3 * ret.natoms();
    ret.n3n = n3n;
    ret.i3n3n = n3n * (n3n + 1) * (n3n + 2) / 6;
    ret.i4n3n = n3n * (n3n + 1) * (n3n + 2) * (n3n + 3) / 24;
    let nvib = n3n - 6 + usize::from(ret.is_linear());
    ret.nvib = nvib;
    ret.i2vib = ioff(nvib + 1);
    ret.i3vib = nvib * (nvib + 1) * (nvib + 2) / 6;
    ret.i4vib = nvib * (nvib + 1) * (nvib + 2) * (nvib + 3) / 24;
}

/// read `filename` and return part of a `Spectro`
fn read<P>(filename: P) -> Spectro
where
    P: AsRef<Path> + Debug + Clone,
{
    let f = match File::open(filename.clone()) {
        Ok(f) => f,
        Err(_) => {
            eprintln!("failed to open infile '{filename:?}'");
            std::process::exit(1);
        }
    };
    let reader = BufReader::new(f);
    enum State {
        Header,
        Geom,
        Weight,
        Curvil,
        Degmode,
        None,
    }
    // map of string coordinates to their atom number
    let mut coord_map = HashMap::new();
    let mut state = State::None;
    let mut skip = 0;
    let mut ret = Spectro::default();
    for line in reader.lines().flatten() {
        if skip > 0 {
            skip -= 1;
        } else if line.contains("SPECTRO") {
            state = State::Header;
        } else if line.contains("GEOM") {
            skip = 1;
            state = State::Geom;
        } else if line.contains("WEIGHT") {
            skip = 1;
            state = State::Weight;
        } else if line.contains("CURVIL") {
            state = State::Curvil;
        } else if line.contains("DEGMODE") {
            state = State::Degmode;
        } else {
            match state {
                State::Header => {
                    ret.header.extend(
                        line.split_whitespace()
                            .map(|s| s.parse::<usize>().unwrap()),
                    );
                }
                State::Geom => {
                    let mut fields = line.split_whitespace();
                    let atomic_number =
                        fields.next().unwrap().parse::<f64>().unwrap() as usize;
                    // collect after nexting off the first value
                    let fields: Vec<_> = fields.collect();
                    match atomic_number {
                        0 => {
                            let mut dummy_coords = Vec::new();
                            for coord in fields {
                                dummy_coords.push(
                                    if let Some(idx) = coord_map.get(coord) {
                                        DummyVal::Atom(*idx)
                                    } else {
                                        DummyVal::Value(coord.parse().unwrap())
                                    },
                                );
                            }
                            ret.dummies.push(Dummy::from(dummy_coords));
                        }
                        _ => {
                            let atom_index = ret.geom.atoms.len();
                            for coord in &fields {
                                // don't mind overwriting another atom
                                // because that means their coordinates are
                                // the same
                                coord_map.insert(coord.to_string(), atom_index);
                            }
                            ret.geom.atoms.push(Atom {
                                atomic_number,
                                x: fields[0].parse().unwrap(),
                                y: fields[1].parse().unwrap(),
                                z: fields[2].parse().unwrap(),
                            });
                        }
                    }
                }
                State::Weight => {
                    let fields = line.split_whitespace().collect::<Vec<_>>();
                    ret.weights.push((
                        fields[0].parse::<usize>().unwrap(),
                        fields[1].parse::<f64>().unwrap(),
                    ));
                }
                State::Curvil => {
                    use crate::Curvil::*;
                    let v = parse_line(&line);
                    // TODO differentiate between other curvils with 4
                    // coordinates. probably by requiring them to be written
                    // out like in intder so they don't have to be specified
                    // at the top too
                    match v.len() {
                        2 => ret.curvils.push(Bond(v[0], v[1])),
                        3 => ret.curvils.push(Bend(v[0], v[1], v[2])),
                        4 => ret.curvils.push(Tors(v[0], v[1], v[2], v[3])),
                        0 => (),
                        _ => todo!("unrecognized number of curvils"),
                    }
                }
                State::Degmode => {
                    let v = parse_line(&line);
                    if !v.is_empty() {
                        ret.degmodes.push(v);
                    }
                }
                State::None => (),
            }
        }
    }
    ret
}

pub fn load_fc2<P>(infile: P, n3n: usize) -> Dmat
where
    P: AsRef<Path>,
{
    let data = read_to_string(infile).unwrap();
    Dmat::from_iterator(
        n3n,
        n3n,
        data.split_whitespace().map(|s| s.parse().unwrap()),
    )
}

/// load a Tensor3 from `file` in SPECTRO format
pub fn load_fc3<P>(infile: P, n3n: usize) -> Tensor3
where
    P: AsRef<Path> + std::fmt::Debug,
{
    let f33 = load_vec(infile);
    new_fc3(n3n, &f33)
}

/// create a Tensor3 ready to use in `run` from a slice of cubic force
/// constants
pub fn new_fc3(n3n: usize, f33: &[f64]) -> tensor::Tensor3<f64> {
    let mut f3x = Tensor3::zeros(n3n, n3n, n3n);
    let mut labc = 0;
    for iabc in 0..n3n {
        for jabc in 0..=iabc {
            for kabc in 0..=jabc {
                let val = f33[labc];
                f3x[(iabc, jabc, kabc)] = val;
                f3x[(iabc, kabc, jabc)] = val;
                f3x[(jabc, iabc, kabc)] = val;
                f3x[(jabc, kabc, iabc)] = val;
                f3x[(kabc, iabc, jabc)] = val;
                f3x[(kabc, jabc, iabc)] = val;
                labc += 1;
            }
        }
    }
    f3x
}

/// load a Tensor4 from `file` in SPECTRO format
pub fn load_fc4<P>(infile: P, n3n: usize) -> Tensor4
where
    P: AsRef<Path> + std::fmt::Debug,
{
    let f44 = load_vec(infile);
    new_fc4(n3n, &f44)
}

/// create a Tensor4 ready to use in `run` from a slice of quartic force
/// constants
pub fn new_fc4(n3n: usize, f44: &[f64]) -> Tensor4 {
    let mut f4x = Tensor4::zeros(n3n, n3n, n3n, n3n);
    let mut mabc = 0;
    for iabc in 0..n3n {
        for jabc in 0..=iabc {
            for kabc in 0..=jabc {
                for labc in 0..=kabc {
                    let val = f44[mabc];
                    f4x[(iabc, jabc, kabc, labc)] = val;
                    f4x[(iabc, jabc, labc, kabc)] = val;
                    f4x[(iabc, kabc, jabc, labc)] = val;
                    f4x[(iabc, kabc, labc, jabc)] = val;
                    f4x[(iabc, labc, jabc, kabc)] = val;
                    f4x[(iabc, labc, kabc, jabc)] = val;
                    f4x[(jabc, iabc, kabc, labc)] = val;
                    f4x[(jabc, iabc, labc, kabc)] = val;
                    f4x[(jabc, kabc, iabc, labc)] = val;
                    f4x[(jabc, kabc, labc, iabc)] = val;
                    f4x[(jabc, labc, iabc, kabc)] = val;
                    f4x[(jabc, labc, kabc, iabc)] = val;
                    f4x[(kabc, iabc, jabc, labc)] = val;
                    f4x[(kabc, iabc, labc, jabc)] = val;
                    f4x[(kabc, jabc, iabc, labc)] = val;
                    f4x[(kabc, jabc, labc, iabc)] = val;
                    f4x[(kabc, labc, iabc, jabc)] = val;
                    f4x[(kabc, labc, jabc, iabc)] = val;
                    f4x[(labc, iabc, jabc, kabc)] = val;
                    f4x[(labc, iabc, kabc, jabc)] = val;
                    f4x[(labc, jabc, iabc, kabc)] = val;
                    f4x[(labc, jabc, kabc, iabc)] = val;
                    f4x[(labc, kabc, iabc, jabc)] = val;
                    f4x[(labc, kabc, jabc, iabc)] = val;
                    mabc += 1;
                }
            }
        }
    }
    f4x
}

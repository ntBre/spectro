use std::{
    collections::HashMap,
    fmt::Debug,
    fs::File,
    io::{BufRead, BufReader},
    path::Path,
};

use crate::utils::{linalg::symm_eigen_decomp3, *};
use crate::Spectro;
use crate::{
    consts::CONST,
    dummy::{Dummy, DummyVal},
};
use symm::Atom;

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

/// perform the geometry manipulations from dist.f on ret.geom and set the
/// corresponding fields in `ret`
pub(crate) fn process_geom(ret: &mut Spectro) {
    // assumes input geometry in bohr
    ret.geom.to_angstrom();
    let com = ret.geom.com();
    ret.geom.translate(-com);
    let moi = ret.geom.moi();
    let (pr, mut axes) = symm_eigen_decomp3(moi);
    ret.primat = Vec::from(pr.as_slice());
    ret.rotcon = pr.iter().map(|m| CONST / m).collect();
    ret.rotor = ret.rotor_type(&pr);
    if ret.rotor.is_sym_top() {
        const TOL: f64 = 1e-5;
        let iaxis = if close(pr[0], pr[1], TOL) {
            3
        } else if close(pr[0], pr[2], TOL) {
            2
        } else if close(pr[1], pr[2], TOL) {
            1
        } else {
            panic!("not a symmetric top");
        };

        if iaxis == 1 {
            let egr = nalgebra::matrix![
            0.0, 0.0, -1.0;
            0.0, 1.0,  0.0;
            1.0, 0.0,  0.0;
            ];
            let atemp = axes * egr;
            axes = atemp;
            let pb1 = CONST / pr[(2)];
            let pb2 = CONST / pr[(1)];
            let pb3 = CONST / pr[(0)];
            ret.primat[(0)] = pr[(2)];
            ret.primat[(1)] = pr[(1)];
            ret.primat[(2)] = pr[(0)];
            ret.rotcon[(0)] = pb1;
            ret.rotcon[(1)] = pb2;
            ret.rotcon[(2)] = pb3;
        } else if iaxis == 2 {
            todo!("dist.f:419");
        }
    }
    ret.geom = ret.geom.transform(axes.transpose());
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
    if ret.rotor.is_sym_top() {
        // NOTE smallest eps for which this works for the current test cases
        const TOL: f64 = 1e-6;
        let pg = ret.geom.point_group_approx(TOL);
        use symm::PointGroup::*;
        let iatl = match pg {
            C1 => todo!(),
            C2 { axis: _ } => todo!(),
            Cs { plane: _ } => todo!(),
            C2v { axis: _, planes: _ } => todo!(),
            C3v { axis, plane } => {
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
            D2h { axes: _, planes: _ } => todo!(),
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
        };
    }
    ret.natom = ret.natoms();
    let n3n = 3 * ret.natoms();
    ret.n3n = n3n;
    ret.i3n3n = n3n * (n3n + 1) * (n3n + 2) / 6;
    ret.i4n3n = n3n * (n3n + 1) * (n3n + 2) * (n3n + 3) / 24;
    let nvib = n3n - 6 + if ret.is_linear() { 1 } else { 0 };
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
            eprintln!("failed to open infile '{:?}'", filename);
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

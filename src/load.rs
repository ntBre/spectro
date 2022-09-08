use std::{
    cmp::{max, min},
    collections::{HashMap, HashSet},
    fmt::Debug,
    fs::File,
    io::{BufRead, BufReader, Result},
    path::Path,
};

use crate::f3qcm::F3qcm;
use crate::f4qcm::F4qcm;
use crate::ifrm1::Ifrm1;
use crate::ifrm2::Ifrm2;
use crate::quartic::Quartic;
use crate::resonance::{Coriolis, Fermi1, Fermi2, Restst};
use crate::rot::Rot;
use crate::rotor::{Rotor, ROTOR_EPS};
use crate::sextic::Sextic;
use crate::state::State;
use crate::utils::*;
use crate::{
    consts::CONST,
    dummy::{Dummy, DummyVal},
};
use crate::{
    consts::{ALPHA_CONST, FACT2},
    Spectro,
};
use nalgebra::DMatrix;
use symm::{Atom, Molecule};
use tensor::Tensor4;

impl Spectro {
    pub fn load<P>(filename: P) -> Self
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
                            fields.next().unwrap().parse::<f64>().unwrap()
                                as usize;
                        // collect after nexting off the first value
                        let fields: Vec<_> = fields.collect();
                        match atomic_number {
                            0 => {
                                let mut dummy_coords = Vec::new();
                                for coord in fields {
                                    dummy_coords.push(
                                        if let Some(idx) = coord_map.get(coord)
                                        {
                                            DummyVal::Atom(*idx)
                                        } else {
                                            DummyVal::Value(
                                                coord.parse().unwrap(),
                                            )
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
                                    coord_map
                                        .insert(coord.to_string(), atom_index);
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
                        let fields =
                            line.split_whitespace().collect::<Vec<_>>();
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
        // assumes input geometry in bohr
        ret.geom.to_angstrom();
        let (pr, axes) = ret.geom.normalize();
        ret.primat = Vec::from(pr.as_slice());
        ret.rotcon = pr.iter().map(|m| CONST / m).collect();
        ret.axes = axes;
        ret.rotor = ret.rotor_type(&pr);
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
        ret
    }
}

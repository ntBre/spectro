use std::{
    collections::HashMap,
    fmt::Debug,
    fs::{read_to_string, File},
    io::{BufRead, BufReader, Result},
};

use intder::DMat;

mod dummy;
use dummy::{Dummy, DummyVal};

mod utils;
use nalgebra::Matrix3;
use rotor::{Rotor, ROTOR_EPS};
use utils::*;

mod rotor;

use symm::{Atom, Molecule};

#[cfg(test)]
mod tests;

type Mat3 = Matrix3<f64>;

#[derive(Default, Clone, Debug, PartialEq)]
pub struct Spectro {
    pub header: Vec<usize>,
    pub geom: Molecule,
    pub weights: Vec<(usize, f64)>,
    pub curvils: Vec<Vec<usize>>,
    pub degmodes: Vec<Vec<usize>>,
    pub dummies: Vec<Dummy>,
}

impl Spectro {
    pub fn load(filename: &str) -> Self {
        let f = match File::open(filename) {
            Ok(f) => f,
            Err(_) => {
                eprintln!("failed to open infile '{}'", filename);
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
                        let v = parse_line(&line);
                        if !v.is_empty() {
                            ret.curvils.push(v);
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

    pub fn write(&self, filename: &str) -> Result<()> {
        use std::io::Write;
        let mut f = File::create(filename)?;
        writeln!(f, "{}", self)?;
        Ok(())
    }

    /// compute the type of molecular rotor in `self.geom` assuming it has
    /// already been normalized and reordered. These tests are taken from the
    /// [Crawford Programming
    /// Projects](https://github.com/CrawfordGroup/ProgrammingProjects/blob/master/Project%2301/hints/step7-solution.md)
    fn rotor_type(&self) -> Rotor {
        if self.geom.atoms.len() == 2 {
            return Rotor::Diatomic;
        }
        let close = |a, b| f64::abs(a - b) < ROTOR_EPS;
        let moms = self.geom.principal_moments();
        if moms[0] < ROTOR_EPS {
            Rotor::Linear
        } else if close(moms[0], moms[1]) && close(moms[1], moms[2]) {
            Rotor::SphericalTop
        } else if close(moms[0], moms[1]) && !close(moms[1], moms[2]) {
            Rotor::OblateSymmTop
        } else if !close(moms[0], moms[1]) && close(moms[1], moms[2]) {
            Rotor::ProlateSymmTop
        } else {
            Rotor::AsymmTop
        }
    }

    pub fn natoms(&self) -> usize {
        self.geom.atoms.len()
    }

    // run spectro
    pub fn run(mut self) {
        // assumes input geometry in bohr
        self.geom.to_angstrom();

        self.geom.normalize();
        let axes = self.geom.reorder();

        println!("{}", axes);

        let rotor = self.rotor_type();
        println!("Molecule is {}", rotor);

        let n3n = 3 * self.natoms();

        let fc2 = load_fc2("testfiles/fort.15", n3n);
        let fc2 = self.rot2nd(fc2, axes);
    }

    /// rotate the force constants in `fx` to align with the principal axes in
    /// `eg` used to align the geometry
    pub fn rot2nd(&self, fx: DMat, eg: Mat3) -> DMat {
        let mut ret = fx.clone();
        let natom = self.natoms();
        for ia in (0..3 * natom).step_by(3) {
            for ib in (0..3 * natom).step_by(3) {
                // grab 3x3 blocks of FX into A, perform Eg × A × Egᵀ and set
                // that block in the return matrix
                let a = fx.fixed_slice::<3, 3>(ia, ib);
                let temp2 = (eg * a) * eg.transpose();
                let mut targ = ret.fixed_slice_mut::<3, 3>(ia, ib);
                targ.copy_from(&temp2);
            }
        }
        ret
    }
}

pub fn load_fc2(infile: &str, n3n: usize) -> DMat {
    let data = read_to_string(infile).unwrap();
    DMat::from_iterator(
        n3n,
        n3n,
        data.split_whitespace().map(|s| s.parse().unwrap()),
    )
}

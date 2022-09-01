#![feature(test)]

use std::{
    collections::{HashMap, HashSet},
    f64::consts::PI,
    fmt::Debug,
    fs::File,
    io::{BufRead, BufReader, Result},
    path::Path,
};

mod dummy;
use dummy::{Dummy, DummyVal};

mod utils;
use nalgebra::DMatrix;
use resonance::{Coriolis, Darling, Fermi1, Fermi2};
use rot::Rot;
use rotor::{Rotor, ROTOR_EPS};
use sextic::Sextic;
use tensor::Tensor4;
type Tensor3 = tensor::tensor3::Tensor3<f64>;
use utils::*;
mod rot;

mod resonance;
mod rotor;

use symm::{Atom, Molecule};

#[cfg(test)]
mod tests;

type Mat3 = nalgebra::Matrix3<f64>;
type Dvec = nalgebra::DVector<f64>;
type Dmat = nalgebra::DMatrix<f64>;

const _HE: f64 = 4.359813653;
const _A0: f64 = 0.52917706;
/// HE / (AO * AO) from fortran. something about hartrees and AO is bohr radius
const FACT2: f64 = 4.359813653 / (0.52917706 * 0.52917706);
const FUNIT3: f64 = 4.359813653 / (0.52917706 * 0.52917706 * 0.52917706);
const FUNIT4: f64 =
    4.359813653 / (0.52917706 * 0.52917706 * 0.52917706 * 0.52917706);
/// pre-computed √(elmass/amu)
const _FAC1: f64 = 0.02342178947039116194;
const _AMU: f64 = 1.66056559e-27;
const _ELMASS: f64 = 0.91095344e-30;
/// looks like cm-1 to mhz factor
const CL: f64 = 2.99792458;
const ALAM: f64 = 4.0e-2 * (PI * PI * CL) / (PH * AVN);
/// constant for converting moments of inertia to rotational constants
const CONST: f64 = 1.0e+02 * (PH * AVN) / (8.0e+00 * PI * PI * CL);
/// planck's constant in atomic units?

const PH: f64 = 6.626176;
/// pre-computed sqrt of ALAM
const SQLAM: f64 = 0.17222125037910759882;
const FACT3: f64 = 1.0e6 / (SQLAM * SQLAM * SQLAM * PH * CL);
const FACT4: f64 = 1.0e6 / (ALAM * ALAM * PH * CL);

/// PRINCIPAL ---> CARTESIAN
static IPTOC: nalgebra::Matrix3x6<usize> = nalgebra::matrix![
2,0,1,1,2,0;
0,1,2,2,1,0;
0,2,1,1,0,2;
];

/// CARTESIAN---> PRINCIPAL
static ICTOP: nalgebra::Matrix3x6<usize> = nalgebra::matrix![
    1,2,0,2,0,1;
    0,1,2,2,1,0;
    0,2,1,1,0,2;
];

/// avogadro's number
const AVN: f64 = 6.022045;
const _PARA: f64 = 1.0 / AVN;
// pre-compute the sqrt and make const
const SQRT_AVN: f64 = 2.4539855337796920273026076438896;
// conversion to cm-1
const WAVE: f64 = 1e4 * SQRT_AVN / (2.0 * PI * CL);

#[derive(Clone, Debug, PartialEq)]
pub enum Curvil {
    Bond(usize, usize),

    Bend(usize, usize, usize),

    Tors(usize, usize, usize, usize),
}

pub mod quartic;
use quartic::*;
pub mod sextic;

/// struct containing the fields to describe a Spectro input file:
/// ```text
/// header: Vec<usize>: the input options
/// geom: Molecule: the geometry
/// weights: Vec<(usize, f64)>: atom index - weight pairs
/// curvils: Vec<Curvil>: curvilinear coordinates
/// degmodes: Vec<Vec<usize>>: degenerate modes
/// dummies: Vec<Dummy>: dummy atoms
/// ```
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Spectro {
    pub header: Vec<usize>,
    pub geom: Molecule,
    pub weights: Vec<(usize, f64)>,
    pub curvils: Vec<Curvil>,
    pub degmodes: Vec<Vec<usize>>,
    pub dummies: Vec<Dummy>,
    pub rotor: Rotor,
    pub n3n: usize,
    pub i3n3n: usize,
    pub i4n3n: usize,
    pub nvib: usize,
    pub i2vib: usize,
    pub i3vib: usize,
    pub i4vib: usize,
    pub natom: usize,
    pub axes: Mat3,
    pub primat: Vec<f64>,
    pub rotcon: Vec<f64>,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct Restst {
    pub coriolis: Vec<Coriolis>,
    pub fermi1: Vec<Fermi1>,
    pub fermi2: Vec<Fermi2>,
    pub darling: Vec<Darling>,
    pub states: Vec<State>,
    pub modes: Vec<Mode>,
}

#[derive(Clone, Debug, PartialEq)]
pub enum Mode {
    I1(usize),
    I2(usize, usize),
    I3(usize, usize, usize),
}

impl Mode {
    pub fn count(modes: &[Self]) -> (usize, usize, usize) {
        let mut ret = (0, 0, 0);
        for m in modes {
            match m {
                Mode::I1(_) => ret.0 += 1,
                Mode::I2(_, _) => ret.1 += 1,
                Mode::I3(_, _, _) => ret.2 += 1,
            }
        }
        ret
    }

    pub fn partition(
        modes: &[Self],
    ) -> (Vec<usize>, Vec<(usize, usize)>, Vec<usize>) {
        let mut ret = (vec![], vec![], vec![]);
        for m in modes {
            match m {
                &Mode::I1(i) => ret.0.push(i),
                &Mode::I2(i, j) => {
                    ret.1.push((i, j));
                }
                &Mode::I3(i, j, k) => {
                    ret.2.push(i);
                    ret.2.push(j);
                    ret.2.push(k);
                }
            }
        }
        ret
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum State {
    /// singly-degenerate mode state
    I1st(Vec<usize>),

    /// doubly-degenerate mode state
    I2st(Vec<usize>),

    /// triply-degenerate mode state
    I3st(Vec<usize>),

    /// combination band of a singly-degenerate mode and a doubly-degenerate
    /// mode
    I12st { i1st: Box<State>, i2st: Box<State> },
}

impl Spectro {
    pub fn is_linear(&self) -> bool {
        self.rotor == Rotor::Linear
    }

    /// return a ready-to-use spectro without a template
    pub fn nocurvil() -> Self {
        Self {
            // only important fields are 1=Ncart to ignore curvils, 2=Isotop to
            // use default weights, 8=Nderiv to do a QFF, and 21=Iaverg to get
            // vibrationally averaged coordinates (that one might not be
            // important)
            header: vec![
                99, 1, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
                0, 0, 0, 0, 0, 0, 0, 0, 0,
            ],
            ..Self::default()
        }
    }

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
        ret.rotor = ret.rotor_type();
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
        let moms = &self.primat;
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

    /// Calculate the zeta, big A and Wilson A and J matrices. Zeta is for the
    /// coriolis coupling constants
    fn zeta(&self, lxm: &Dmat, w: &[f64]) -> (Tensor3, Dmat) {
        let zmat = make_zmat(self.nvib, self.natom, lxm);
        // calculate the A vectors. says only half is formed since it's
        // symmetric
        let mut wila = Dmat::zeros(self.nvib, 6);
        for k in 0..self.nvib {
            let mut valuxx = 0.0;
            let mut valuyy = 0.0;
            let mut valuzz = 0.0;
            let mut valuxy = 0.0;
            let mut valuxz = 0.0;
            let mut valuyz = 0.0;
            for i in 0..self.natom {
                let ix = 3 * i;
                let iy = ix + 1;
                let iz = iy + 1;
                let xcm = self.geom.atoms[i].x;
                let ycm = self.geom.atoms[i].y;
                let zcm = self.geom.atoms[i].z;
                let rmass = w[i].sqrt();
                valuxx += rmass * (ycm * lxm[(iy, k)] + zcm * lxm[(iz, k)]);
                valuyy += rmass * (xcm * lxm[(ix, k)] + zcm * lxm[(iz, k)]);
                valuzz += rmass * (xcm * lxm[(ix, k)] + ycm * lxm[(iy, k)]);
                valuxy -= rmass * xcm * lxm[(iy, k)];
                valuxz -= rmass * xcm * lxm[(iz, k)];
                valuyz -= rmass * ycm * lxm[(iz, k)];
            }
            wila[(k, 0)] = 2.0 * valuxx;
            wila[(k, 1)] = 2.0 * valuxy;
            wila[(k, 2)] = 2.0 * valuyy;
            wila[(k, 3)] = 2.0 * valuxz;
            wila[(k, 4)] = 2.0 * valuyz;
            wila[(k, 5)] = 2.0 * valuzz;
        }
        (zmat, wila)
    }

    /// formation of the secular equation
    pub fn form_sec(&self, fx: Dmat, sqm: &[f64]) -> Dmat {
        let mut fxm = fx.clone();
        for i in 0..self.n3n {
            let ii = i / 3;
            for j in i..self.n3n {
                let jj = j / 3;
                fxm[(i, j)] = sqm[ii] * fx[(i, j)] * sqm[jj];
            }
        }
        fxm.fill_lower_triangle_with_upper_triangle();
        fxm
    }

    /// rotate the harmonic force constants in `fx` to align with the principal
    /// axes in `self.axes` used to align the geometry
    pub fn rot2nd(&self, fx: Dmat) -> Dmat {
        let (a, b) = fx.shape();
        let mut ret = Dmat::zeros(a, b);
        let natom = self.natoms();
        for ia in (0..3 * natom).step_by(3) {
            for ib in (0..3 * natom).step_by(3) {
                // grab 3x3 blocks of FX into A, perform Eg × A × Egᵀ and set
                // that block in the return matrix
                let a = fx.fixed_slice::<3, 3>(ia, ib);
                let temp2 = self.axes.transpose() * a * self.axes;
                let mut targ = ret.fixed_slice_mut::<3, 3>(ia, ib);
                targ.copy_from(&temp2);
            }
        }
        ret
    }

    /// rotate the cubic force constants in `f3x` to align with the principal
    /// axes in `eg` used to align the geometry
    pub fn rot3rd(&self, f3x: Tensor3, eg: Mat3) -> Tensor3 {
        let (a, b, c) = f3x.shape();
        let mut ret = Tensor3::zeros(a, b, c);
        // TODO could try slice impl like in rot2nd, but it will be harder with
        // tensors
        for i in 0..self.n3n {
            for j in 0..self.natom {
                let ib = j * 3;
                for k in 0..self.natom {
                    let ic = k * 3;
                    let mut a = Mat3::zeros();
                    for jj in 0..3 {
                        for kk in 0..3 {
                            a[(jj, kk)] = f3x[(ib + jj, ic + kk, i)];
                        }
                    }
                    let temp2 = eg.transpose() * a * eg;
                    for jj in 0..3 {
                        for kk in 0..3 {
                            ret[(ib + jj, ic + kk, i)] = temp2[(jj, kk)];
                        }
                    }
                }
            }
        }

        for j in 0..self.n3n {
            for k in 0..self.n3n {
                for i in 0..self.natom {
                    let ia = i * 3;
                    let mut val = [0.0; 3];
                    for ii in 0..3 {
                        val[ii] = ret[(j, k, ia)] * eg[(0, ii)]
                            + ret[(j, k, ia + 1)] * eg[(1, ii)]
                            + ret[(j, k, ia + 2)] * eg[(2, ii)];
                    }

                    for ii in 0..3 {
                        ret[(j, k, ia + ii)] = val[ii];
                    }
                }
            }
        }
        ret
    }

    /// rotate the quartic force constants in `f4x` to align with the principal
    /// axes in `eg` used to align the geometry
    pub fn rot4th(&self, f4x: Tensor4, eg: Mat3) -> Tensor4 {
        let egt = eg.transpose();
        let (a, b, c, d) = f4x.shape();
        let mut ret = Tensor4::zeros(a, b, c, d);
        for i in 0..self.n3n {
            for j in 0..self.n3n {
                for k in 0..self.natom {
                    let ic = k * 3;
                    for l in 0..self.natom {
                        let id = l * 3;
                        let mut a = Mat3::zeros();
                        for kk in 0..3 {
                            for ll in 0..3 {
                                a[(kk, ll)] = f4x[(i, j, ic + kk, id + ll)];
                            }
                        }
                        let temp2 = egt * a * eg;
                        for kk in 0..3 {
                            for ll in 0..3 {
                                ret[(i, j, ic + kk, id + ll)] = temp2[(kk, ll)];
                            }
                        }
                    }
                }
            }
        }

        for k in 0..self.n3n {
            for l in 0..self.n3n {
                for i in 0..self.natom {
                    let ia = i * 3;
                    for j in 0..self.natom {
                        let ib = j * 3;
                        let mut a = Mat3::zeros();
                        for ii in 0..3 {
                            for jj in 0..3 {
                                a[(ii, jj)] = ret[(ia + ii, ib + jj, k, l)];
                            }
                        }
                        let temp2 = egt * a * eg;
                        for ii in 0..3 {
                            for jj in 0..3 {
                                ret[(ia + ii, ib + jj, k, l)] = temp2[(ii, jj)];
                            }
                        }
                    }
                }
            }
        }
        ret
    }

    /// make the LX matrix
    fn make_lx(&self, sqm: &[f64], lxm: &Dmat) -> Dmat {
        let mut lx = Dmat::zeros(self.n3n, self.n3n);
        for i in 0..self.n3n {
            let ii = i / 3;
            for j in 0..self.n3n {
                lx[(i, j)] = sqm[ii] * lxm[(i, j)];
            }
        }
        lx
    }

    /// helper method for alpha matrix in `alphaa`
    fn alpha(
        &self,
        freq: &Dvec,
        wila: &Dmat,
        zmat: &Tensor3,
        f3qcm: &[f64],
        coriolis: &[Coriolis],
    ) -> Dmat {
        /// CONST IS THE PI*SQRT(C/H) FACTOR
        const CONST: f64 = 0.086112;
        let mut alpha = Dmat::zeros(self.nvib, 3);
        // NOTE like the fermi resonances, this overwrites earlier resonances.
        // should it use them all?
        let mut icorol: HashMap<(usize, usize), usize> = HashMap::new();
        for &Coriolis { i, j, axis } in coriolis {
            icorol.insert((i, j), axis as usize);
            icorol.insert((j, i), axis as usize);
        }
        for ixyz in 0..3 {
            for i in 0..self.nvib {
                let ii = ioff(ixyz + 2) - 1;
                let valu0 = 2.0 * self.rotcon[ixyz].powi(2) / freq[i];
                let mut valu1 = 0.0;
                for jxyz in 0..3 {
                    let ij = ioff(ixyz.max(jxyz) + 1) + ixyz.min(jxyz);
                    valu1 += wila[(i, ij)].powi(2) / self.primat[jxyz];
                }
                valu1 *= 0.75;

                let mut valu2 = 0.0;
                let mut valu3 = 0.0;
                for j in 0..self.nvib {
                    if j != i {
                        let wisq = freq[i].powi(2);
                        let wjsq = freq[j].powi(2);
                        let tmp = icorol.get(&(i, j));
                        if tmp.is_some() && *tmp.unwrap() == ixyz {
                            valu2 -= 0.5
                                * zmat[(i, j, ixyz)].powi(2)
                                * (freq[i] - freq[j]).powi(2)
                                / (freq[j] * (freq[i] + freq[j]));
                        } else {
                            valu2 += zmat[(i, j, ixyz)].powi(2)
                                * (3.0 * wisq + wjsq)
                                / (wisq - wjsq);
                        }
                    }
                    let wj32 = freq[j].powf(1.5);
                    let iij = find3r(j, i, i);
                    valu3 += wila[(j, ii)] * f3qcm[iij] * freq[i] / wj32;
                }
                alpha[(i, ixyz)] = valu0 * (valu1 + valu2 + valu3 * CONST);
            }
        }
        alpha
    }

    /// compute the vibrationally-averaged rotational constants for asymmetric
    /// tops
    fn alphaa(
        &self,
        rotcon: &[f64],
        freq: &Dvec,
        wila: &Dmat,
        zmat: &Tensor3,
        f3qcm: &[f64],
        fund: &[f64],
        modes: &[Mode],
        states: &[State],
        coriolis: &[Coriolis],
    ) -> Dmat {
        let alpha = self.alpha(freq, &wila, &zmat, f3qcm, coriolis);
        // do the fundamentals + the ground state
        let nstop = fund.len() + 1;
        let n1dm = fund.len();
        let mut rotnst = Dmat::zeros(nstop, 3);
        for axis in 0..3 {
            for ist in 0..nstop {
                let mut suma = 0.0;
                for ii in 0..n1dm {
                    let i = match modes[ii] {
                        Mode::I1(i) => i,
                        Mode::I2(_, _) => todo!(),
                        Mode::I3(_, _, _) => todo!(),
                    };
                    match &states[ist] {
                        State::I1st(v) => {
                            suma += alpha[(i, axis)] * (v[ii] as f64 + 0.5);
                        }
                        State::I2st(_) => todo!(),
                        State::I3st(_) => todo!(),
                        State::I12st { i1st: _, i2st: _ } => todo!(),
                    }
                }
                let bva = rotcon[axis] + suma;
                rotnst[(ist, axis)] = bva;
            }
        }
        rotnst
    }

    /// rotational energy levels of an asymmmetric top
    fn rota(
        &self,
        rotnst: &Dmat,
        states: &[State],
        quartic: &Quartic,
    ) -> Vec<Rot> {
        let (b4a, b5a, b6a) = quartic.arots();

        // I think this is always 3, but that doesn't really make sense. set
        // to 0 in mains, then passed to readw which never touches it and
        // then set to 3 if it's still 0. there might also be a better way
        // to set these than maxk. check what they actually loop over
        // NOTE: pretty sure this is always the case
        let irep = 0;
        let (ic, _) = princ_cart(irep);
        // number of states here is just the ground state + fundamentals,
        // singly-vibrationally excited states, but changes with resonances
        // TODO take this from actual i1sts when I have that. currently i1sts is
        // all states, but I want to partition it to the singly-degenerate
        // states.

        // use the nstop determined earlier
        let (nstop, _) = rotnst.shape();
        let nderiv = self.header[7];
        // this is a 600 line loop fml
        let mut ret = Vec::new();
        for nst in 0..nstop {
            // this is inside a conditional in the fortran code, but it
            // would be really annoying to return these from inside it here
            assert!(nderiv > 2);
            let vib1 = rotnst[(nst, 0)] - self.rotcon[(0)];
            let vib2 = rotnst[(nst, 1)] - self.rotcon[(1)];
            let vib3 = rotnst[(nst, 2)] - self.rotcon[(2)];
            let mut vibr = [0.0; 3];
            vibr[ic[0]] = vib1;
            vibr[ic[1]] = vib2;
            vibr[ic[2]] = vib3;
            let bxa = b4a + vibr[0];
            let bya = b5a + vibr[1];
            let bza = b6a + vibr[2];
            match &states[nst] {
                State::I1st(v) => {
                    ret.push(Rot::new(v.clone(), bza, bxa, bya));
                }
                State::I2st(_) => todo!(),
                State::I3st(_) => todo!(),
                State::I12st { i1st: _, i2st: _ } => todo!(),
            }
            // TODO return these S ones too
            // let bxs = b1s + vibr[0];
            // let bys = b2s + vibr[1];
            // let bzs = b3s + vibr[2];
        }
        ret
    }

    pub fn run<P>(self, fort15: P, fort30: P, fort40: P) -> Output
    where
        P: AsRef<Path>,
    {
        // load the force constants, rotate them to the new axes, and convert
        // them to the proper units
        let fc2 = load_fc2(fort15, self.n3n);
        let fc2 = self.rot2nd(fc2);
        let fc2 = FACT2 * fc2;

        // form the secular equations and decompose them to get harmonic
        // frequencies and the LXM matrix
        let w = self.geom.weights();
        let sqm: Vec<_> = w.iter().map(|w| 1.0 / w.sqrt()).collect();
        let fxm = self.form_sec(fc2, &sqm);
        let (harms, lxm) = symm_eigen_decomp(fxm);
        let freq = to_wavenumbers(harms);

        // form the LX matrix
        let lx = self.make_lx(&sqm, &lxm);

        // start of cubic analysis
        let f3x = load_fc3(fort30, self.n3n);
        let mut f3x = self.rot3rd(f3x, self.axes);
        let f3qcm = force3(self.n3n, &mut f3x, &lx, self.nvib, &freq);

        // start of quartic analysis
        let f4x = load_fc4(fort40, self.n3n);
        let f4x = self.rot4th(f4x, self.axes);
        let f4qcm = force4(self.n3n, &f4x, &lx, self.nvib, &freq);

        let (zmat, wila) = self.zeta(&lxm, &w);

        let Restst {
            coriolis,
            fermi1,
            fermi2,
            darling: _,
            states,
            modes,
        } = self.restst(&zmat, &f3qcm, &freq);

        let (xcnst, e0) = xcalc(
            self.nvib,
            &f4qcm,
            &freq,
            &f3qcm,
            &zmat,
            &self.rotcon,
            &fermi1,
            &fermi2,
        );

        let funds = make_funds(&freq, self.nvib, &xcnst);

        // TODO good to here
        let rotnst = self.alphaa(
            &self.rotcon,
            &freq,
            &wila,
            &zmat,
            &f3qcm,
            &funds,
            &modes,
            &states,
            &coriolis,
        );

        // this is worked on by resona and then enrgy so keep it out here
        let nstate = states.len();
        let mut eng = vec![0.0; nstate];

        resona(e0, &modes, &freq, &xcnst, &fermi1, &fermi2, &mut eng);

        enrgy(
            &funds, &freq, &xcnst, &f3qcm, e0, &states, &modes, &fermi1,
            &fermi2, &mut eng,
        );

        let mut corrs = Vec::new();
        let mut harms = Vec::new();
        for i in 1..self.nvib + 1 {
            corrs.push(eng[i] - eng[0]);
            harms.push(freq[i - 1]);
        }

        // print_vib_states(&eng, &states);

        let quartic = Quartic::new(&self, &freq, &wila);
        let _sextic = Sextic::new(&self, &wila, &zmat, &freq, &f3qcm);
        let rots = self.rota(&rotnst, &states, &quartic);

        Output {
            harms: Dvec::from(harms),
            funds,
            rots,
            corrs,
        }
    }

    // should return all of the resonances, as well as the states (I think)
    pub(crate) fn restst(
        &self,
        zmat: &Tensor3,
        f3qcm: &[f64],
        freq: &Dvec,
    ) -> Restst {
        let mut modes = Vec::new();
        // probably I could get rid of this if, but I guess it protects against
        // accidental degmodes in asymmetric tops. is an accident still
        // degenerate?
        use Mode::*;
        if self.rotor.is_sym_top() {
            const DEG_TOL: f64 = 0.1;
            let mut triples = HashSet::new();
            for i in 0..self.nvib {
                let fi = freq[i];
                for j in i + 1..self.nvib {
                    let fj = freq[j];
                    for k in j + 1..self.nvib {
                        let fk = freq[k];
                        if close(fi, fj, DEG_TOL) && close(fi, fk, DEG_TOL) {
                            triples.insert(i);
                            triples.insert(j);
                            triples.insert(k);
                            modes.push(I3(i, j, k));
                        }
                    }
                }
            }
            // run these in two passes to avoid counting something as
            // triply-degenerate AND doubly-degenerate
            let mut doubles = HashSet::new();
            for i in 0..self.nvib {
                let fi = freq[i];
                for j in i + 1..self.nvib {
                    let fj = freq[j];
                    if close(fi, fj, DEG_TOL)
                        && !triples.contains(&i)
                        && !triples.contains(&j)
                    {
                        doubles.insert(i);
                        doubles.insert(j);
                        modes.push(I2(i, j));
                    }
                }
            }
            for i in 0..self.nvib {
                if !triples.contains(&i) && !doubles.contains(&i) {
                    modes.push(I1(i));
                }
            }
        } else {
            for i in 0..self.nvib {
                modes.push(I1(i));
            }
        };
        let coriolis = self.rotor.coriolis(&modes, freq, zmat);
        let fermi1 = self.rotor.fermi1(&modes, freq, f3qcm);
        let fermi2 = self.rotor.fermi2(&modes, freq, f3qcm);
        let darling = self.rotor.darling(&modes, freq);

        let (n1dm, n2dm, n3dm) = Mode::count(&modes);
        if n3dm > 0 {
            todo!("untested");
        }

        let mut states = Vec::new();
        use State::*;

        // ground state, all zeros
        states.push(I1st(vec![0; self.nvib]));

        // fundamentals, single excitations
        for ii in 0..n1dm {
            let mut tmp = vec![0; self.nvib];
            tmp[ii] = 1;
            states.push(I1st(tmp));
        }
        for ii in 0..n2dm {
            let mut tmp = vec![0; self.nvib];
            tmp[ii] = 1;
            states.push(I2st(tmp));
        }
        for ii in 0..n3dm {
            let mut tmp = vec![0; self.nvib];
            tmp[ii] = 1;
            states.push(I3st(tmp));
        }
        // NOTE: triply-degenerate modes are only handled for fundamentals. also
        // their resonances are not handled at all

        // overtones, double excitations in one mode
        for ii in 0..n1dm {
            let mut tmp = vec![0; self.nvib];
            tmp[ii] = 2;
            states.push(I1st(tmp));
        }
        for ii in 0..n2dm {
            let mut tmp = vec![0; self.nvib];
            tmp[ii] = 2;
            states.push(I2st(tmp));
        }

        // combination bands
        for ii in 0..n1dm {
            // nondeg - nondeg combination
            for jj in ii + 1..n1dm {
                let mut tmp = vec![0; self.nvib];
                tmp[ii] = 1;
                tmp[jj] = 1;
                states.push(I1st(tmp));
            }

            // nondeg - deg combination
            for jj in 0..n2dm {
                // I guess you push this to states as well. might have to change
                // how I index these because they set states(ii, ist) to this,
                // which I guess syncs it with ist in i2sts. we'll see how it's
                // accessed. I'm not that interested in combination bands anyway
                let mut i1st = vec![0; self.nvib];
                i1st[ii] = 1;
                let mut i2st = vec![0; self.nvib];
                i2st[jj] = 1;
                states.push(I12st {
                    i1st: Box::new(I1st(i1st)),
                    i2st: Box::new(I2st(i2st)),
                })
            }
        }

        // deg-deg combination
        for ii in 0..n2dm {
            for jj in ii + 1..n2dm {
                let mut tmp = vec![0; self.nvib];
                tmp[ii] = 1;
                tmp[jj] = 1;
                states.push(I2st(tmp));
            }
        }

        Restst {
            coriolis,
            fermi1,
            fermi2,
            darling,
            states,
            modes,
        }
    }
}

/// build the zeta matrix
fn make_zmat(nvib: usize, natom: usize, lxm: &Dmat) -> tensor::Tensor3<f64> {
    let mut zmat = Tensor3::zeros(nvib, nvib, 3);
    for k in 0..nvib {
        for l in 0..nvib {
            let mut valux = 0.0;
            let mut valuy = 0.0;
            let mut valuz = 0.0;
            for i in 0..natom {
                let ix = 3 * i;
                let iy = ix + 1;
                let iz = iy + 1;
                valux +=
                    lxm[(iy, k)] * lxm[(iz, l)] - lxm[(iz, k)] * lxm[(iy, l)];
                valuy +=
                    lxm[(iz, k)] * lxm[(ix, l)] - lxm[(ix, k)] * lxm[(iz, l)];
                valuz +=
                    lxm[(ix, k)] * lxm[(iy, l)] - lxm[(iy, k)] * lxm[(ix, l)];
            }
            zmat[(k, l, 0)] = valux;
            zmat[(k, l, 1)] = valuy;
            zmat[(k, l, 2)] = valuz;
        }
    }
    zmat
}

fn resona(
    e0: f64,
    modes: &[Mode],
    freq: &Dvec,
    xcnst: &Dmat,
    fermi1: &Vec<Fermi1>,
    fermi2: &Vec<Fermi2>,
    eng: &mut [f64],
) {
    let (n1dm, _, _) = Mode::count(modes);
    let (i1mode, _, _) = Mode::partition(modes);
    let mut zpe = e0;
    for ii in 0..n1dm {
        let i = i1mode[ii];
        zpe += freq[i] * 0.5;
        for jj in 0..=ii {
            let j = i1mode[jj];
            zpe += xcnst[(i, j)] * 0.25;
        }
    }
    let iirst = make_resin(fermi1, n1dm, fermi2);
    let (nreson, _) = iirst.shape();
    for ist in 0..nreson {
        let mut e = e0;
        for ii in 0..n1dm {
            let i = i1mode[ii];
            e += freq[i] * (iirst[(ist, ii)] as f64 + 0.5);
            for jj in 0..=ii {
                let j = i1mode[jj];
                e += xcnst[(i, j)]
                    * (iirst[(ist, ii)] as f64 + 0.5)
                    * (iirst[(ist, jj)] as f64 + 0.5);
            }
        }
        eng[ist] = e - zpe;
    }
}

/// construct the RESIN Fermi polyad matrix. NOTE that the comments in resona.f
/// mention multiple blocks for different symmetries. However, the only way we
/// use it is with a single block, so I'm writing this code with that in mind.
fn make_resin(
    fermi1: &Vec<Fermi1>,
    n1dm: usize,
    fermi2: &Vec<Fermi2>,
) -> DMatrix<usize> {
    let mut data: HashSet<Vec<usize>> = HashSet::new();
    for &Fermi1 { i, j } in fermi1 {
        // 2wi
        let mut tmp = vec![0; n1dm];
        tmp[i] = 2;
        data.insert(tmp);
        // = wj
        let mut tmp = vec![0; n1dm];
        tmp[j] = 1;
        data.insert(tmp);
    }
    for &Fermi2 { i, j, k } in fermi2 {
        // wi + wj
        let mut tmp = vec![0; n1dm];
        tmp[i] = 1;
        tmp[j] = 1;
        data.insert(tmp);
        // = wk
        let mut tmp = vec![0; n1dm];
        tmp[k] = 1;
        data.insert(tmp);
    }
    let data: Vec<_> = data.iter().cloned().flatten().collect();
    let resin =
        DMatrix::<usize>::from_row_slice(data.len() / n1dm, n1dm, &data);
    resin
}

/// contains all of the output data from running Spectro
#[derive(Clone, Debug)]
pub struct Output {
    /// harmonic frequencies
    pub harms: Dvec,

    /// partially resonance-corrected anharmonic frequencies
    pub funds: Vec<f64>,

    /// fully resonance-corrected anharmonic frequencies
    pub corrs: Vec<f64>,

    /// vibrationally averaged rotational constants
    pub rots: Vec<Rot>,
}

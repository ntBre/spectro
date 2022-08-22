use std::{
    collections::HashMap,
    f64::consts::PI,
    fmt::Debug,
    fs::File,
    io::{BufRead, BufReader, Result},
};

use intder::DMat;

mod dummy;
use dummy::{Dummy, DummyVal};

mod utils;
use rot::Rot;
use rotor::{Rotor, ROTOR_EPS};
use tensor::{Tensor3, Tensor4};
use utils::*;
mod rot;

mod rotor;

use symm::{Atom, Molecule};

#[cfg(test)]
mod tests;

type Mat3 = nalgebra::Matrix3<f64>;
type Vec3 = nalgebra::Vector3<f64>;
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
use sextic::*;

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
    pub rotor_type: Rotor,
    pub n3n: usize,
    pub i3n3n: usize,
    pub i4n3n: usize,
    pub nvib: usize,
    pub i2vib: usize,
    pub i3vib: usize,
    pub i4vib: usize,
    pub natom: usize,
    pub axes: Mat3,
}

impl Spectro {
    pub fn is_linear(&self) -> bool {
        self.rotor_type == Rotor::Linear
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
        ret.geom.normalize();
        ret.axes = ret.geom.reorder();
        ret.rotor_type = ret.rotor_type();
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

    /// Calculate the zeta, big A and Wilson A and J matrices. Zeta is for the
    /// coriolis coupling constants
    fn zeta(
        &self,
        natom: usize,
        nvib: usize,
        lxm: &Dmat,
        w: &[f64],
    ) -> (Tensor3, Tensor3, Dmat) {
        // calculate the zeta matrix for the coriolis coupling constants
        let mut zmat = Tensor3::zeros(nvib, nvib, 3);
        let mut biga = Tensor3::zeros(nvib, nvib, 6);
        for k in 0..nvib {
            for l in 0..nvib {
                let mut valux = 0.0;
                let mut valuy = 0.0;
                let mut valuz = 0.0;
                let mut valuxx = 0.0;
                let mut valuyy = 0.0;
                let mut valuxy = 0.0;
                let mut valuxz = 0.0;
                let mut valuyz = 0.0;
                let mut valuzz = 0.0;
                for i in 0..natom {
                    let ix = 3 * i;
                    let iy = ix + 1;
                    let iz = iy + 1;
                    // TODO use add assign here after writing a test. also
                    // consider adding directly to the tensors instead of these
                    // intermediate values. might be a lot of array accesses,
                    // but it simplifies the code at least. a benchmark might
                    // help decide
                    valux = valux + lxm[(iy, k)] * lxm[(iz, l)]
                        - lxm[(iz, k)] * lxm[(iy, l)];
                    valuy = valuy + lxm[(iz, k)] * lxm[(ix, l)]
                        - lxm[(ix, k)] * lxm[(iz, l)];
                    valuz = valuz + lxm[(ix, k)] * lxm[(iy, l)]
                        - lxm[(iy, k)] * lxm[(ix, l)];
                    valuxx = valuxx
                        + lxm[(iy, k)] * lxm[(iy, l)]
                        + lxm[(iz, k)] * lxm[(iz, l)];
                    valuyy = valuyy
                        + lxm[(ix, k)] * lxm[(ix, l)]
                        + lxm[(iz, k)] * lxm[(iz, l)];
                    valuzz = valuzz
                        + lxm[(ix, k)] * lxm[(ix, l)]
                        + lxm[(iy, k)] * lxm[(iy, l)];
                    valuxy = valuxy - lxm[(ix, k)] * lxm[(iy, l)];
                    valuxz = valuxz - lxm[(ix, k)] * lxm[(iz, l)];
                    valuyz = valuyz - lxm[(iy, k)] * lxm[(iz, l)];
                }
                zmat[(k, l, 0)] = valux;
                zmat[(k, l, 1)] = valuy;
                zmat[(k, l, 2)] = valuz;
                biga[(k, l, 0)] = valuxx;
                biga[(k, l, 1)] = valuxy;
                biga[(k, l, 2)] = valuyy;
                biga[(k, l, 3)] = valuxz;
                biga[(k, l, 4)] = valuyz;
                biga[(k, l, 5)] = valuzz;
            }
        }
        let mut wila = Dmat::zeros(nvib, 6);
        // calculate the A vectors. says only half is formed since it's
        // symmetric
        for k in 0..nvib {
            let mut valuxx = 0.0;
            let mut valuyy = 0.0;
            let mut valuzz = 0.0;
            let mut valuxy = 0.0;
            let mut valuxz = 0.0;
            let mut valuyx = 0.0;
            let mut valuzx = 0.0;
            let mut valuzy = 0.0;
            let mut valuyz = 0.0;
            for i in 0..natom {
                let ix = 3 * i;
                let iy = ix + 1;
                let iz = iy + 1;
                let xcm = self.geom.atoms[i].x;
                let ycm = self.geom.atoms[i].y;
                let zcm = self.geom.atoms[i].z;
                let rmass = w[i].sqrt();
                valuxx =
                    valuxx + rmass * (ycm * lxm[(iy, k)] + zcm * lxm[(iz, k)]);
                valuyy =
                    valuyy + rmass * (xcm * lxm[(ix, k)] + zcm * lxm[(iz, k)]);
                valuzz =
                    valuzz + rmass * (xcm * lxm[(ix, k)] + ycm * lxm[(iy, k)]);
                valuxy = valuxy - rmass * xcm * lxm[(iy, k)];
                valuxz = valuxz - rmass * xcm * lxm[(iz, k)];
                valuyz = valuyz - rmass * ycm * lxm[(iz, k)];
                valuyx = valuyx - rmass * ycm * lxm[(ix, k)];
                valuzx = valuzx - rmass * zcm * lxm[(ix, k)];
                valuzy = valuzy - rmass * zcm * lxm[(iy, k)];
            }
            wila[(k, 0)] = 2.0 * valuxx;
            wila[(k, 1)] = 2.0 * valuxy;
            wila[(k, 2)] = 2.0 * valuyy;
            wila[(k, 3)] = 2.0 * valuxz;
            wila[(k, 4)] = 2.0 * valuyz;
            wila[(k, 5)] = 2.0 * valuzz;
        }
        (zmat, biga, wila)
    }

    /// formation of the secular equation
    pub fn form_sec(&self, fx: DMat, n3n: usize, sqm: &[f64]) -> DMat {
        let mut fxm = fx.clone();
        for i in 0..n3n {
            let ii = i / 3;
            for j in i..n3n {
                let jj = j / 3;
                fxm[(i, j)] = sqm[ii] * fx[(i, j)] * sqm[jj];
            }
        }
        fxm.fill_lower_triangle_with_upper_triangle();
        fxm
    }

    /// rotate the harmonic force constants in `fx` to align with the principal
    /// axes in `eg` used to align the geometry
    pub fn rot2nd(&self, fx: DMat, eg: Mat3) -> DMat {
        let mut ret = fx.clone();
        let natom = self.natoms();
        for ia in (0..3 * natom).step_by(3) {
            for ib in (0..3 * natom).step_by(3) {
                // grab 3x3 blocks of FX into A, perform Eg × A × Egᵀ and set
                // that block in the return matrix
                let a = fx.fixed_slice::<3, 3>(ia, ib);
                let temp2 = eg * a * eg.transpose();
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
                    let temp2 = eg * a * eg.transpose();
                    for jj in 0..3 {
                        for kk in 0..3 {
                            ret[(ib + jj, ic + kk, i)] = temp2[(jj, kk)];
                        }
                    }
                }
            }
        }

        // have to use the transpose to get the same indices as the fortran
        // version
        let eg = eg.transpose();
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
                        let temp2 = eg * a * eg.transpose();
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
                        let temp2 = eg * a * eg.transpose();
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
    fn make_lx(&self, n3n: usize, sqm: &[f64], lxm: &Dmat) -> Dmat {
        let mut lx = Dmat::zeros(n3n, n3n);
        for i in 0..n3n {
            let ii = i / 3;
            for j in 0..n3n {
                lx[(i, j)] = sqm[ii] * lxm[(i, j)];
            }
        }
        lx
    }

    /// compute the vibrationally-averaged rotational constants for asymmetric
    /// tops
    fn alphaa(
        &self,
        nvib: usize,
        rotcon: &[f64],
        freq: &Dvec,
        wila: &Dmat,
        zmat: &Tensor3,
        f3qcm: &[f64],
        fund: &[f64],
        i1mode: &[usize],
        i1sts: &Vec<Vec<usize>>,
    ) -> Dmat {
        let primat = self.geom.principal_moments();
        let alpha = alpha(nvib, rotcon, freq, &wila, &primat, &zmat, f3qcm);
        let nstop = 4;
        let n1dm = fund.len();
        let mut rotnst = Dmat::zeros(nstop, 3);
        for axis in 0..3 {
            for ist in 0..nstop {
                let mut suma = 0.0;
                for ii in 0..n1dm {
                    let i = i1mode[ii];
                    suma += alpha[(i, axis)] * (i1sts[ist][ii] as f64 + 0.5);
                }
                let bva = rotcon[axis] + suma;
                rotnst[(ist, axis)] = bva;
            }
        }
        rotnst
    }

    /// rotational energy levels of an asymmmetric top
    #[allow(unused)]
    fn rota(
        &self,
        rotnst: &Dmat,
        i1sts: &Vec<Vec<usize>>,
        rotcon: &[f64],
        quartic: &Quartic,
        sextic: &Sextic,
    ) -> Vec<Rot> {
        let Quartic {
            sigma: _,
            rkappa: _,
            delj,
            delk,
            deljk,
            sdelk,
            sdelj,
            bxa: b4a,
            bya: b5a,
            bza: b6a,
            djn: _,
            djkn: _,
            dkn: _,
            sdjn: _,
            r5: _,
            r6: _,
            dj: _,
            djk: _,
            dk: _,
            sd1: _,
            sd2: _,
            bxs: b1s,
            bys: b2s,
            bzs: b3s,
            djw: _,
            djkw: _,
            dkw: _,
        } = quartic;
        let Sextic {
            phij,
            phijk,
            phikj,
            phik,
            sphij,
            sphijk,
            sphik,
        } = sextic;

        // I think this is always 3, but that doesn't really make sense. set
        // to 0 in mains, then passed to readw which never touches it and
        // then set to 3 if it's still 0. there might also be a better way
        // to set these than maxk. check what they actually loop over
        let maxj = 3;
        let maxk = 2 * maxj + 1;
        // NOTE: pretty sure this is always the case
        let irep = 0;
        let (ic, _) = princ_cart(irep);
        // number of states here is just the ground state + fundamentals,
        // singly-vibrationally excited states, but changes with resonances
        // TODO take this from actual i1sts when I have that. currently i1sts is
        // all states, but I want to partition it to the singly-degenerate
        // states.
        let nstop = 4;
        let nderiv = self.header[7];
        // this is a 600 line loop fml
        let mut ret = Vec::new();
        for nst in 0..nstop {
            // this is inside a conditional in the fortran code, but it
            // would be really annoying to return these from inside it here
            assert!(nderiv > 2);
            let vib1 = rotnst[(nst, 0)] - rotcon[(0)];
            let vib2 = rotnst[(nst, 1)] - rotcon[(1)];
            let vib3 = rotnst[(nst, 2)] - rotcon[(2)];
            let mut vibr = [0.0; 3];
            vibr[ic[0]] = vib1;
            vibr[ic[1]] = vib2;
            vibr[ic[2]] = vib3;
            let bxa = b4a + vibr[0];
            let bya = b5a + vibr[1];
            let bza = b6a + vibr[2];
            ret.push(Rot::new(i1sts[nst].clone(), bza, bxa, bya));
            // TODO return these S ones too
            let bxs = b1s + vibr[0];
            let bys = b2s + vibr[1];
            let bzs = b3s + vibr[2];
        }
        ret
    }

    pub fn run(self) -> Output {
        let moments = self.geom.principal_moments();
        let rotcon: Vec<_> = moments.iter().map(|m| CONST / m).collect();
        let natom = self.natoms();

        // load the force constants, rotate them to the new axes, and convert
        // them to the proper units
        let fc2 = load_fc2("testfiles/fort.15", self.n3n);
        let fc2 = self.rot2nd(fc2, self.axes);
        let fc2 = FACT2 * fc2;

        // form the secular equations and decompose them to get harmonic
        // frequencies and the LXM matrix
        let w = self.geom.weights();
        let sqm: Vec<_> = w.iter().map(|w| 1.0 / w.sqrt()).collect();
        let fxm = self.form_sec(fc2, self.n3n, &sqm);
        let (harms, lxm) = symm_eigen_decomp(fxm);
        let freq = to_wavenumbers(harms);

        // form the LX matrix, not used so far
        let lx = self.make_lx(self.n3n, &sqm, &lxm);

        // not used yet
        let (zmat, _biga, wila) = self.zeta(natom, self.nvib, &lxm, &w);

        let quartic = Quartic::new(&self, self.nvib, &freq, &wila, &rotcon);

        // start of cubic analysis
        let f3x = load_fc3("testfiles/fort.30", self.n3n);
        let mut f3x = self.rot3rd(f3x, self.axes);
        let f3qcm =
            force3(self.n3n, &mut f3x, &lx, self.nvib, &freq, self.i3vib);

        // start of quartic analysis
        let f4x = load_fc4("testfiles/fort.40", self.n3n);
        let mut f4x = self.rot4th(f4x, self.axes);
        let f4qcm =
            force4(self.n3n, &mut f4x, &lx, self.nvib, &freq, self.i4vib);

        // TODO RESTST

        // TODO get this from RESTST
        // NOTE: right now these are *not* i1sts, they are all of the states.
        // i1sts is really only the first four
        let i1sts = vec![
            vec![0, 0, 0],
            vec![1, 0, 0],
            vec![0, 1, 0],
            vec![0, 0, 1],
            vec![2, 0, 0],
            vec![0, 2, 0],
            vec![0, 0, 2],
            vec![1, 1, 0],
            vec![1, 0, 1],
            vec![0, 1, 1],
        ];

        // TODO get this from RESTST
        let i1mode = vec![0, 1, 2];

        // end ALPHAA

        let sextic = Sextic::new(&self, &wila, &zmat, &freq, &f3qcm, &rotcon);

        let (xcnst, e0) =
            xcalc(self.nvib, &f4qcm, &freq, &f3qcm, &zmat, &rotcon);

        let fund = funds(&freq, self.nvib, &xcnst);

        let rotnst = self.alphaa(
            self.nvib, &rotcon, &freq, &wila, &zmat, &f3qcm, &fund, &i1mode,
            &i1sts,
        );

        let _reng = enrgy(&fund, &freq, &xcnst, e0, &i1sts, &i1mode);

        let rots = self.rota(&rotnst, &i1sts, &rotcon, &quartic, &sextic);

        Output {
            harms: freq,
            funds: fund,
            rots,
        }
    }
}

/// contains all of the output data from running Spectro
pub struct Output {
    /// harmonic frequencies
    pub harms: Dvec,

    /// non-resonance-corrected anharmonic frequencies
    pub funds: Vec<f64>,

    /// vibrationally averaged rotational constants
    pub rots: Vec<Rot>,
}

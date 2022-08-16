#![allow(unused)]
use std::{
    collections::HashMap,
    f64::consts::PI,
    fmt::Debug,
    fs::{read_to_string, File},
    io::{BufRead, BufReader, Result},
};

use intder::DMat;

mod dummy;
use dummy::{Dummy, DummyVal};

mod utils;
use nalgebra::{Matrix3, SymmetricEigen};
use rotor::{Rotor, ROTOR_EPS};
use tensor::{Tensor3, Tensor4};
use utils::*;

mod rotor;

use symm::{Atom, Molecule};

#[cfg(test)]
mod tests;

type Mat3 = Matrix3<f64>;
type Dvec = nalgebra::DVector<f64>;
type Dmat = nalgebra::DMatrix<f64>;

/// HE / (AO * AO) from fortran. something about hartrees and AO is bohr radius
const FACT2: f64 = 4.359813653 / (0.52917706 * 0.52917706);
const FUNIT3: f64 = 4.359813653 / (0.52917706 * 0.52917706 * 0.52917706);
/// looks like cm-1 to mhz factor
const CL: f64 = 2.99792458;
const ALAM: f64 = 4.0e-2 * (PI * PI * CL) / (PH * AVN);
/// planck's constant in atomic units?
const PH: f64 = 6.626176;
/// pre-computed sqrt of ALAM
const SQLAM: f64 = 0.17222125037910759882;
const FACT3: f64 = 1.0e6 / (SQLAM * SQLAM * SQLAM * PH * CL);
/// avogadro's number
const AVN: f64 = 6.022045;
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
    pub is_linear: Option<bool>,
}

impl Spectro {
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

        let rotor = self.rotor_type();
        println!("Molecule is {}", rotor);

        let natom = self.natoms();
        let n3n = 3 * natom;
        let i3n3n = n3n * (n3n + 1) * (n3n + 2) / 6;
        let i4n3n = n3n * (n3n + 1) * (n3n + 2) * (n3n + 3) / 24;

        let nvib = n3n - 6
            + if let Rotor::Linear = rotor {
                self.is_linear = Some(true);
                1
            } else {
                self.is_linear = Some(false);
                0
            };
        let i2vib = ioff(nvib + 1);
        let i3vib = nvib * (nvib + 1) * (nvib + 2) / 6;
        let i4vib = nvib * (nvib + 1) * (nvib + 2) * (nvib + 3) / 24;

        // load the force constants, rotate them to the new axes, and convert
        // them to the proper units
        let fc2 = load_fc2("testfiles/fort.15", n3n);
        let fc2 = self.rot2nd(fc2, axes);
        let fc2 = FACT2 * fc2;

        // form the secular equations and decompose them to get harmonic
        // frequencies and the LXM matrix
        let w = self.geom.weights();
        let sqm: Vec<_> = w.iter().map(|w| 1.0 / w.sqrt()).collect();
        let fxm = self.form_sec(fc2, n3n, &sqm);
        let (harms, lxm) = symm_eigen_decomp(fxm);
        let harms = to_wavenumbers(harms);

        // form the LX matrix, not used so far
        let lx = self.make_lx(n3n, &sqm, &lxm);

        // not used yet
        let (zmat, biga, wila) = self.zeta(natom, nvib, &lxm, &w);

        // only for the quartic distortion coefficients, doesn't appear to touch
        // anything else
        self.qcent(nvib, &harms, &wila);

        // start of cubic analysis
        let f3x = load_fc3("testfiles/fort.30", n3n);
        let mut f3x = self.rot3rd(n3n, natom, f3x, axes);

	// TODO not sure what is supposed to come out of this yet
        force3(n3n, &mut f3x, lx, nvib, harms, i3vib);
    }

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

    /// calculate the quartic centrifugal distortion constants. TODO this will
    /// be the first test for the output of zeta
    fn qcent(&self, nvib: usize, freq: &Dvec, wila: &Dmat) {
        // convert to cm-1 from the biggest mess you've ever seen
        const CONST1: f64 = 3.833384078e04;
        // convert to mhz from cm-1
        const CONST2: f64 = 2.99792458e04;

        let maxcor = if self.is_linear.unwrap() { 2 } else { 3 };
        let primat = self.geom.principal_moments();

        let mut tau = Tensor4::zeros(maxcor, maxcor, maxcor, maxcor);
        for ixyz in 0..maxcor {
            for jxyz in 0..maxcor {
                for kxyz in 0..maxcor {
                    for lxyz in 0..maxcor {
                        let ijxyz = ioff(ixyz.max(jxyz) + 1) + ixyz.min(jxyz);
                        let klxyz = ioff(kxyz.max(lxyz) + 1) + kxyz.min(lxyz);
                        let mut sum = 0.0;
                        for k in 0..nvib {
                            let div = freq[k].powi(2)
                                * primat[ixyz]
                                * primat[jxyz]
                                * primat[kxyz]
                                * primat[lxyz];
                            sum += wila[(k, ijxyz)] * wila[(k, klxyz)] / div;
                        }
                        tau[(ixyz, jxyz, kxyz, lxyz)] = -0.5 * CONST1 * sum;
                        // println!(
                        //     "{ixyz:5}{jxyz:5}{kxyz:5}{lxyz:5}{:15.8}",
                        //     tau[(ixyz, jxyz, kxyz, lxyz)]
                        // );
                    }
                }
            }
        }

        // form tau prime in wavenumbers
        let mut taupcm = Dmat::zeros(maxcor, maxcor);
        for ijxyz in 0..maxcor {
            for klxyz in 0..maxcor {
                taupcm[(ijxyz, klxyz)] = tau[(ijxyz, ijxyz, klxyz, klxyz)];
                if ijxyz != klxyz {
                    taupcm[(ijxyz, klxyz)] +=
                        2.0 * tau[(ijxyz, klxyz, ijxyz, klxyz)];
                }
            }
        }
        // TODO resume at line 175 of qcent.f
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

    /// rotate the cubic force constants in `fx` to align with the principal
    /// axes in `eg` used to align the geometry
    pub fn rot3rd(
        &self,
        n3n: usize,
        natom: usize,
        f3x: Tensor3,
        eg: Mat3,
    ) -> Tensor3 {
        let (a, b, c) = f3x.shape();
        let mut ret = Tensor3::zeros(a, b, c);
        // TODO could try slice impl like in rot2nd, but it will be harder with
        // tensors
        for i in 0..n3n {
            for j in 0..natom {
                let ib = j * 3;
                for k in 0..natom {
                    let ic = k * 3;
                    let mut a = Matrix3::zeros();
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
        for j in 0..n3n {
            for k in 0..n3n {
                for i in 0..natom {
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
}

fn force3(
    n3n: usize,
    f3x: &mut Tensor3,
    lx: Dmat,
    nvib: usize,
    harms: Dvec,
    i3vib: usize,
) {
    let mut dd = Dmat::zeros(n3n, n3n);
    for kabc in 0..n3n {
        for i in 0..n3n {
            for j in 0..n3n {
                dd[(i, j)] = f3x[(i, j, kabc)];
            }
        }
        dd *= FUNIT3;
        let ee = lx.clone().transpose() * dd.clone() * lx.clone();
        for i in 0..n3n {
            for j in 0..n3n {
                f3x[(i, j, kabc)] = ee[(i, j)];
            }
        }
    }
    let mut f3q = Tensor3::zeros(n3n, n3n, n3n);
    for i in 0..n3n {
        for j in 0..n3n {
            for k in 0..n3n {
                let mut val = 0.0;
                for l in 0..n3n {
                    val += f3x[(i, j, l)] * lx[(l, k)];
                }
                f3q[(i, j, k)] = val;
            }
        }
    }
    let mut frq3 = Tensor3::zeros(nvib, nvib, nvib);
    for ivib in 0..nvib {
        let wk = harms[ivib];
        for ii in 0..nvib {
            let wi = harms[ii];
            for jj in 0..nvib {
                let wj = harms[jj];
                let wijk = wi * wj * wk;
                let sqws = wijk.sqrt();
                let fact = FACT3 / sqws;
                dd[(ii, jj)] = f3q[(ii, jj, ivib)] * fact;
                frq3[(ii, jj, ivib)] = f3q[(ii, jj, ivib)];
            }
        }
    }
    // NOTE skipped a loop above this and after it that looked like unit
    // manipulation. might be the same case here if facts3 is never used
    // elsewhere
    let mut facts3 = vec![1.0; i3vib];
    for i in 0..nvib {
        let iii = find3r(i, i, i);
        facts3[iii] = 6.0;
        // intentionally i-1
        for j in 0..i {
            let iij = find3r(i, i, j);
            let ijj = find3r(i, j, j);
            facts3[iij] = 2.0;
            facts3[ijj] = 2.0;
        }
    }
}

/// cubic force constant indexing formula. I think it relies on the fortran
/// numbering though, so I need to add one initially and then subtract one at
/// the end
fn find3r(i: usize, j: usize, k: usize) -> usize {
    let i = i + 1;
    let j = j + 1;
    let k = k + 1;
    (i - 1) * i * (i + 1) / 6 + (j - 1) * j / 2 + k - 1
}

pub fn ioff(n: usize) -> usize {
    let mut sum = 0;
    for i in 0..n {
        sum += i;
    }
    sum
}

/// convert harmonic frequencies to wavenumbers. not sure if this works on
/// anything else
pub fn to_wavenumbers(freqs: Dvec) -> Dvec {
    Dvec::from_iterator(
        freqs.len(),
        freqs.iter().map(|f| {
            if *f < 0.0 {
                -1.0 * WAVE * f64::sqrt(-f)
            } else {
                WAVE * f64::sqrt(*f)
            }
        }),
    )
}

/// compute the eigen decomposition of the symmetric matrix `mat` and return
/// both the sorted eigenvalues and the corresponding eigenvectors in descending
/// order
pub fn symm_eigen_decomp(mat: Dmat) -> (Dvec, Dmat) {
    let SymmetricEigen {
        eigenvectors: vecs,
        eigenvalues: vals,
    } = SymmetricEigen::new(mat);
    let mut pairs: Vec<_> = vals.iter().enumerate().collect();
    pairs.sort_by(|(_, a), (_, b)| b.partial_cmp(&a).unwrap());
    let (rows, cols) = vecs.shape();
    let mut ret = Dmat::zeros(rows, cols);
    for i in 0..cols {
        ret.set_column(i, &vecs.column(pairs[i].0));
    }
    (
        Dvec::from_iterator(vals.len(), pairs.iter().map(|a| a.1.clone())),
        ret,
    )
}

pub fn load_fc2(infile: &str, n3n: usize) -> DMat {
    let data = read_to_string(infile).unwrap();
    DMat::from_iterator(
        n3n,
        n3n,
        data.split_whitespace().map(|s| s.parse().unwrap()),
    )
}

pub fn load_fc3(infile: &str, n3n: usize) -> Tensor3 {
    let mut f3x = Tensor3::zeros(n3n, n3n, n3n);
    let f33 = load_fc34("testfiles/fort.30");
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

pub fn load_fc34(infile: &str) -> Vec<f64> {
    let data = read_to_string(infile).unwrap();
    data.split_whitespace()
        .map(|s| s.parse().unwrap())
        .collect()
}

#![feature(test)]

use std::{
    cmp::{max, min},
    collections::{HashMap, HashSet},
    f64::consts::PI,
    fmt::Debug,
    fs::File,
    io::{BufRead, BufReader, Result},
    path::Path,
};

use dummy::{Dummy, DummyVal};
use f3qcm::F3qcm;
use f4qcm::F4qcm;
use ifrm1::Ifrm1;
use nalgebra::DMatrix;
use quartic::Quartic;
use resonance::{Coriolis, Fermi1, Fermi2, Restst};
use rot::Rot;
use rotor::{Rotor, ROTOR_EPS};
use sextic::Sextic;
use state::State;
use symm::{Atom, Molecule};
use tensor::Tensor4;
use utils::*;

mod dummy;
mod f3qcm;
mod f4qcm;
mod ifrm1;
mod quartic;
mod resonance;
mod rot;
mod rotor;
mod sextic;
mod state;
mod utils;

#[cfg(test)]
mod tests;

type Tensor3 = tensor::tensor3::Tensor3<f64>;
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

/// ALPHA_CONST IS THE PI*SQRT(C/H) FACTOR
const ALPHA_CONST: f64 = 0.086112;

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

#[derive(Clone, Debug, PartialEq)]
pub enum Mode {
    I1(usize),
    I2(usize, usize),
    I3(usize, usize, usize),
}

impl Mode {
    /// return the count of each type of mode in `modes`. these are referred to
    /// in the Fortran code as `n1dm`, `n2dm`, and `n3dm`
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

    /// return vectors of the separated singly-degenerate, doubly-degenerate,
    /// and triply-degenerate modes. these are referrred to in the Fortran code
    /// as `i1mode`, `i2mode`, and `i3mode`.
    pub fn partition(
        modes: &[Self],
    ) -> (Vec<usize>, Vec<(usize, usize)>, Vec<(usize, usize, usize)>) {
        let mut ret = (vec![], vec![], vec![]);
        for m in modes {
            match m {
                &Mode::I1(i) => ret.0.push(i),
                &Mode::I2(i, j) => {
                    ret.1.push((i, j));
                }
                &Mode::I3(i, j, k) => {
                    ret.2.push((i, j, k));
                }
            }
        }
        ret
    }
}

impl Spectro {
    /// helper method for alpha matrix in `alphaa`
    fn alpha(
        &self,
        freq: &Dvec,
        wila: &Dmat,
        zmat: &Tensor3,
        f3qcm: &F3qcm,
        coriolis: &[Coriolis],
    ) -> Dmat {
        let mut alpha = Dmat::zeros(self.nvib, 3);
        let icorol = make_icorol(coriolis);
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
                    valu3 += wila[(j, ii)] * f3qcm[(i, i, j)] * freq[i] / wj32;
                }
                alpha[(i, ixyz)] =
                    valu0 * (valu1 + valu2 + valu3 * ALPHA_CONST);
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
        f3qcm: &F3qcm,
        modes: &[Mode],
        states: &[State],
        coriolis: &[Coriolis],
    ) -> Dmat {
        let alpha = self.alpha(freq, &wila, &zmat, f3qcm, coriolis);
        // do the fundamentals + the ground state
        let nstop = self.nvib + 1;
        let (n1dm, _, _) = Mode::count(modes);
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

    /// compute the vibrationally-averaged rotational constants for symmetric
    /// tops
    fn alphas(
        &self,
        rotcon: &[f64],
        freq: &Dvec,
        wila: &Dmat,
        zmat: &Tensor3,
        f3qcm: &F3qcm,
        modes: &[Mode],
        states: &[State],
        coriolis: &[Coriolis],
    ) -> Dmat {
        if let Rotor::SphericalTop = self.rotor {
            todo!();
        }
        let (ia, ib, ic, iaia, iaib, ibib, ibic) =
            if let Rotor::ProlateSymmTop = self.rotor {
                (2, 1, 0, 5, 3, 0, 1)
            } else {
                (2, 1, 0, 5, 4, 2, 1)
            };
        let icorol = make_icorol(coriolis);
        let (n1dm, n2dm, _) = Mode::count(modes);
        let (i1mode, i2mode, _) = Mode::partition(modes);
        let mut alpha = Dmat::zeros(self.nvib, 3);
        // alpha A
        for kk in 0..n1dm {
            let k = i1mode[kk];
            let valu0 = 2.0 * self.rotcon[ia].powi(2) / freq[k];
            let valu1 = 0.75 * wila[(k, iaia)].powi(2) / self.primat[ia];

            let mut valu2 = 0.0;
            let mut valu3 = 0.0;
            for jj in 0..n1dm {
                let j = i1mode[jj];
                if j != k {
                    let wksq = freq[k].powi(2);
                    let wjsq = freq[j].powi(2);
                    let tmp = icorol.get(&(j, k));
                    if tmp.is_some() && *tmp.unwrap() == ia {
                        valu2 -= 0.5
                            * zmat[(j, k, ia)].powi(2)
                            * (freq[k] - freq[j]).powi(2)
                            / (freq[j] * (freq[k] + freq[j]));
                    } else {
                        valu2 += zmat[(j, k, ia)].powi(2) * (3.0 * wksq + wjsq)
                            / (wksq - wjsq);
                    }
                }
                let wj32 = freq[j].powf(1.5);
                valu3 += wila[(j, iaia)] * f3qcm[(j, k, k)] * freq[k] / wj32;
            }
            alpha[(k, ia)] = valu0 * (valu1 + valu2 + ALPHA_CONST * valu3);
        }
        for kk in 0..n2dm {
            let k = i2mode[kk].0;
            let valu0 = 2.0 * self.rotcon[ia].powi(2) / freq[k];
            let valu1 = 0.75 * wila[(k, iaib)].powi(2) / self.primat[ib];

            let mut valu2 = 0.0;
            let mut valu3 = 0.0;
            for jj in 0..n2dm {
                let (j1, j2) = i2mode[jj];
                if jj == kk {
                    continue;
                }
                let wksq = freq[k].powi(2);
                let wjsq = freq[j1].powi(2);
                let tmp = icorol.get(&(j1, k));
                if tmp.is_some() && *tmp.unwrap() == ia {
                    valu2 -= 0.5
                        * zmat[(k, j2, ia)].powi(2)
                        * (freq[k] - freq[j1]).powi(2)
                        / (freq[j1] * (freq[k] + freq[j1]));
                } else {
                    valu2 += zmat[(k, j2, ia)].powi(2) * (3.0 * wksq + wjsq)
                        / (wksq - wjsq);
                }
            }
            for jj in 0..n1dm {
                let j = i1mode[jj];
                let wj32 = freq[j].powf(1.5);
                valu3 += wila[(j, iaia)] * f3qcm[(j, k, k)] * freq[k] / wj32;
            }
            alpha[(k, ia)] = valu0 * (valu1 + valu2 + ALPHA_CONST * valu3);
        }
        // alpha B - sadly different enough that I can't loop
        for kk in 0..n1dm {
            let k = i1mode[kk];
            let valu0 = 2.0 * self.rotcon[ib].powi(2) / freq[k];
            let valu1 = 0.75
                * (wila[(k, ibib)].powi(2) + wila[(k, ibic)].powi(2))
                / self.primat[ib];

            let mut valu2 = 0.0;
            for jj in 0..n2dm {
                let j = i2mode[jj].0;
                let wjsq = freq[j].powi(2);
                let wksq = freq[k].powi(2);
                let valu3 = (3.0 * wksq + wjsq) / (wksq - wjsq);
                let tmp = icorol.get(&(j, k));
                if tmp.is_some() && *tmp.unwrap() == ic {
                    valu2 -= 0.5
                        * zmat[(k, j, ic)].powi(2)
                        * (freq[k] - freq[j]).powi(2)
                        / (freq[j] * (freq[k] + freq[j]));
                } else {
                    valu2 += zmat[(k, j, ic)].powi(2) * valu3;
                }
                if tmp.is_some() && *tmp.unwrap() == ib {
                    valu2 -= 0.5
                        * zmat[(k, j, ib)].powi(2)
                        * (freq[k] - freq[j]).powi(2)
                        / (freq[j] * (freq[k] + freq[j]));
                } else {
                    valu2 += zmat[(k, j, ib)].powi(2) * valu3;
                }
            }

            let mut valu4 = 0.0;
            for jj in 0..n1dm {
                let j = i1mode[jj];
                let wj32 = freq[j].powf(1.5);
                valu4 += f3qcm[(j, k, k)] * wila[(j, ibib)] * freq[k] / wj32;
            }
            alpha[(k, ib)] = valu0 * (valu1 + valu2 + ALPHA_CONST * valu4);
        }
        for kk in 0..n2dm {
            let (k, _) = i2mode[kk];
            let valu0 = 2.0 * self.rotcon[ib].powi(2) / freq[k];
            // TODO do diatomics count as linear?
            let (valu1, valu2) = if let Rotor::Linear = self.rotor {
                (0.0, 0.0)
            } else {
                let valu1 = 0.375
                    * (2.0 * wila[(k, ibib)].powi(2) / self.primat[ib]
                        + (wila[(k, iaib)].powi(2)) / self.primat[ia]);
                let mut valu2 = 0.0;
                for jj in 0..n2dm {
                    let j = i2mode[jj].0;
                    if jj == kk {
                        continue;
                    }
                    let wjsq = freq[j].powi(2);
                    let wksq = freq[k].powi(2);
                    let valu3 = (3.0 * wksq + wjsq) / (wksq - wjsq);
                    let tmp = icorol.get(&(j, k));
                    if tmp.is_some() && *tmp.unwrap() == ic {
                        valu2 -= 0.5
                            * zmat[(k, j, ic)].powi(2)
                            * (freq[k] - freq[j]).powi(2)
                            / (freq[j] * (freq[k] + freq[j]));
                    } else {
                        valu2 += zmat[(k, j, ic)].powi(2) * valu3;
                    }
                    if tmp.is_some() && *tmp.unwrap() == ib {
                        valu2 -= 0.5
                            * zmat[(k, j, ib)].powi(2)
                            * (freq[k] - freq[j]).powi(2)
                            / (freq[j] * (freq[k] + freq[j]));
                    } else {
                        valu2 += zmat[(k, j, ib)].powi(2) * valu3;
                    }
                }
                (valu1, valu2)
            };

            let mut valu4 = 0.0;
            for jj in 0..n1dm {
                let j = i1mode[jj];
                let wjsq = freq[j].powi(2);
                let wksq = freq[k].powi(2);
                let valu3 = (3.0 * wksq + wjsq) / (wksq - wjsq);
                let tmp = icorol.get(&(j, k));
                if tmp.is_some() && *tmp.unwrap() == ic {
                    valu4 -= 0.5
                        * zmat[(k, j, ic)].powi(2)
                        * (freq[k] - freq[j]).powi(2)
                        / (freq[j] * (freq[k] + freq[j]));
                } else {
                    valu4 += zmat[(k, j, ic)].powi(2) * valu3;
                }
                if tmp.is_some() && *tmp.unwrap() == ib {
                    valu4 -= 0.5
                        * zmat[(k, j, ib)].powi(2)
                        * (freq[k] - freq[j]).powi(2)
                        / (freq[j] * (freq[k] + freq[j]));
                } else {
                    valu4 += zmat[(k, j, ib)].powi(2) * valu3;
                }
            }
            valu4 *= 0.5;

            let mut valu5 = 0.0;
            for jj in 0..n1dm {
                let j = i1mode[jj];
                let wj32 = freq[j].powf(1.5);
                valu5 += f3qcm[(j, k, k)] * wila[(j, ibib)] * freq[k] / wj32;
            }
            alpha[(k, ib)] =
                valu0 * (valu1 + valu2 + valu4 + ALPHA_CONST * valu5);
        }
        // end alpha b

        // vibrationally averaged rotational constants

        // use nstop again to compute only the fundamentals, not every state
        let nstop = self.nvib + 1;
        let mut rotnst = Dmat::zeros(nstop, 3);
        // only one axis for linear molecules
        let axes = if self.rotor.is_linear() {
            vec![ib]
        } else {
            vec![ia, ib]
        };
        let (i1sts, i2sts, _) = State::partition(states);
        for ax in axes {
            for ist in 0..nstop {
                let mut suma = 0.0;
                for ii in 0..n1dm {
                    let i = i1mode[ii];
                    suma += alpha[(i, ax)] * (i1sts[ist][ii] as f64 + 0.5);
                }

                for ii in 0..n2dm {
                    let i = i2mode[ii].0;
                    suma += alpha[(i, ax)] * (i2sts[ist][ii].0 as f64 + 1.0);
                }
                rotnst[(ist, ax)] = rotcon[ax] + suma;
            }
        }
        rotnst
    }

    /// vibrational energy levels and properties in resonance. returns the
    /// energies in the same order as the states in `i1sts`
    fn enrgy(
        &self,
        freq: &Dvec,
        xcnst: &Dmat,
        gcnst: &Option<Dmat>,
        restst: &Restst,
        f3qcm: &F3qcm,
        e0: f64,
        eng: &mut [f64],
    ) {
        let Restst {
            coriolis: _,
            fermi1,
            fermi2,
            darling: _,
            states,
            modes,
            ifunda,
            iovrtn,
            icombn,
        }: &Restst = restst;

        let nstate = states.len();
        let (n1dm, n2dm, _) = Mode::count(modes);
        let (i1mode, i2mode, _) = Mode::partition(modes);
        let (i1sts, i2sts, _) = State::partition(states);
        for nst in 0..nstate {
            let mut val1 = 0.0;
            for (ii, &i) in i1mode.iter().enumerate() {
                val1 += freq[i] * ((i1sts[nst][ii] as f64) + 0.5);
            }

            let mut val2 = 0.0;
            for (ii, &(i, _)) in i2mode.iter().enumerate() {
                val2 += freq[i] * (i2sts[nst][ii].0 as f64 + 1.0);
            }

            // this is val2 in the asym top code
            let mut val3 = 0.0;
            for ii in 0..n1dm {
                let i = i1mode[ii];
                for jj in 0..=ii {
                    let j = i1mode[jj];
                    val3 += xcnst[(i, j)]
                        * ((i1sts[nst][ii] as f64) + 0.5)
                        * ((i1sts[nst][jj] as f64) + 0.5);
                }
            }

            let mut val4 = 0.0;
            for (ii, &i) in i1mode.iter().enumerate() {
                for (jj, &(j, _)) in i2mode.iter().enumerate() {
                    val4 += xcnst[(i, j)]
                        * ((i1sts[nst][ii] as f64 + 0.5)
                            * (i2sts[nst][jj].0 as f64 + 1.0));
                }
            }

            let mut val5 = 0.0;
            for ii in 0..n2dm {
                let i = i2mode[ii].0;
                for jj in 0..=ii {
                    let j = i2mode[jj].0;
                    val5 += xcnst[(i, j)]
                        * ((i2sts[nst][ii].0 as f64 + 1.0)
                            * (i2sts[nst][jj].0 as f64 + 1.0));
                }
            }

            let mut val6 = 0.0;
            for (ii, &(i, _)) in i2mode.iter().enumerate() {
                for (jj, &(j, _)) in i2mode.iter().take(ii + 1).enumerate() {
                    val6 += gcnst
                        .as_ref()
                        .expect("g constants required for symmetric tops")
                        [(i, j)]
                        * (i2sts[nst][ii].1 as f64)
                        * (i2sts[nst][jj].1 as f64);
                }
            }

            eng[nst] = val1 + val2 + val3 + val4 + val5 + val6 + e0;
        }

        let (_, ifrm1) = self.make_fermi_checks(fermi1, fermi2);
        let mut ifrm2: HashMap<(usize, usize), usize> = HashMap::new();
        for f in fermi2 {
            ifrm2.insert((f.i, f.j), f.k);
        }

        if self.rotor.is_sym_top() {
            // NOTE I think this is not going to work at all :( I think my
            // ist/jst stuff inside rsfrm1 is going to break spectacularly here
            // but we'll see
            for ii in 0..n1dm {
                let ivib = i1mode[ii];
                // type 1 fermi resonance
                if let Some(&jvib) = ifrm1.get(&ivib) {
                    let ist = iovrtn[ivib];
                    let jst = ifunda[jvib];
                    rsfrm1(ist, jst, ivib, jvib, f3qcm, eng, false);
                }

                // type 2 fermi resonance
                for jj in ii + 1..n1dm {
                    let jvib = i1mode[jj];
                    if let Some(&kvib) = ifrm2.get(&(jvib, ivib)) {
                        // +1 because that's how I inserted them in restst
                        let ijvib = ioff(max(ivib, jvib) + 1) + min(ivib, jvib);
                        let ijst = icombn[ijvib];
                        let kst = ifunda[kvib];
                        rsfrm2(ijst, kst, ivib, jvib, kvib, f3qcm, eng);
                    }
                }

                for jj in 0..n2dm {
                    let (jvib, _) = i2mode[jj];
                    if let Some(&kvib) = ifrm2.get(&(jvib, ivib)) {
                        let ijvib = ioff(max(ivib, jvib) + 1) + min(ivib, jvib);
                        let ijst = icombn[ijvib];
                        let kst = ifunda[kvib];
                        rsfrm2(ijst, kst, ivib, jvib, kvib, f3qcm, eng);
                    }
                }
            }

            // type 1 again
            for ii in 0..n2dm {
                let (ivib, _) = i2mode[ii];
                if let Some(&jvib) = ifrm1.get(&ivib) {
                    let ist = iovrtn[ivib];
                    let jst = ifunda[jvib];
                    rsfrm1(ist, jst, ivib, jvib, f3qcm, eng, true);
                }

                // type 2 again
                for jj in ii + 1..n2dm {
                    let (jvib, _) = i2mode[jj];
                    if let Some(&kvib) = ifrm2.get(&(jvib, ivib)) {
                        let ijvib = ioff(max(ivib, jvib) + 1) + min(ivib, jvib);
                        let ijst = icombn[ijvib];
                        let kst = ifunda[kvib];
                        rsfrm2(ijst, kst, ivib, jvib, kvib, f3qcm, eng);
                    }
                }
            }
        } else {
            for iii in 0..n1dm {
                let ivib = i1mode[iii];
                if let Some(&jvib) = ifrm1.get(&ivib) {
                    let ist = iovrtn[ivib];
                    let jst = ifunda[jvib];
                    rsfrm1(ist, jst, ivib, jvib, f3qcm, eng, false);
                }
                for jjj in iii + 1..n1dm {
                    let jvib = i1mode[jjj];
                    if let Some(&kvib) = ifrm2.get(&(jvib, ivib)) {
                        let ijvib = ioff(max(ivib, jvib) + 1) + min(ivib, jvib);
                        let ijst = icombn[ijvib];
                        let kst = ifunda[kvib];
                        rsfrm2(ijst, kst, ivib, jvib, kvib, f3qcm, eng);
                    }
                }
            }
        }
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

    pub fn is_linear(&self) -> bool {
        self.rotor == Rotor::Linear
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

    fn make_fermi_checks(
        &self,
        fermi1: &[Fermi1],
        fermi2: &[Fermi2],
    ) -> (tensor::Tensor3<usize>, Ifrm1) {
        let mut ifrmchk = tensor::tensor3::Tensor3::<usize>::zeros(
            self.nvib, self.nvib, self.nvib,
        );
        // using a hash here instead of an array because I need some way to
        // signal that the value is not there. in fortran they use an array of
        // zeros because zero will never be a valid index. I could use -1, but
        // then the vec has to be of isize and I have to do a lot of casting.
        let mut ifrm1 = Ifrm1::new();
        for f in fermi1 {
            ifrmchk[(f.i, f.i, f.j)] = 1;
            ifrm1.insert(f.i, f.j);
        }
        for f in fermi2 {
            ifrmchk[(f.i, f.j, f.k)] = 1;
            ifrmchk[(f.j, f.i, f.k)] = 1;
        }
        (ifrmchk, ifrm1)
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

    pub fn natoms(&self) -> usize {
        self.geom.atoms.len()
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
                _ => (),
                // State::I2st(_) => todo!(),
                // State::I3st(_) => todo!(),
                // State::I12st { i1st: _, i2st: _ } => todo!(),
            }
            // TODO return these S ones too
            // let bxs = b1s + vibr[0];
            // let bys = b2s + vibr[1];
            // let bzs = b3s + vibr[2];
        }
        ret
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
        let freq = to_wavenumbers(&harms);

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

        let restst = Restst::new(&self, &zmat, &f3qcm, &freq);
        let Restst {
            coriolis,
            fermi1,
            fermi2,
            darling: _,
            states,
            modes,
            ifunda: _,
            iovrtn: _,
            icombn: _,
        } = &restst;

        let (xcnst, gcnst, e0) = if self.rotor.is_sym_top() {
            let (x, g, e) = self.xcals(
                &f4qcm, &freq, &f3qcm, &zmat, &fermi1, &fermi2, &modes, &wila,
            );
            (x, Some(g), e)
        } else {
            let (x, e) = self
                .xcalc(&f4qcm, &freq, &f3qcm, &zmat, &modes, &fermi1, &fermi2);
            (x, None, e)
        };

        let (harms, funds) = if self.rotor.is_sym_top() {
            make_sym_funds(&modes, &freq, &xcnst, &gcnst)
        } else {
            (
                freq.as_slice()[..self.nvib].to_vec(),
                make_funds(&freq, self.nvib, &xcnst),
            )
        };

        // TODO good to here
        let rotnst = if self.rotor.is_sym_top() {
            self.alphas(
                &self.rotcon,
                &freq,
                &wila,
                &zmat,
                &f3qcm,
                &modes,
                &states,
                &coriolis,
            )
        } else {
            self.alphaa(
                &self.rotcon,
                &freq,
                &wila,
                &zmat,
                &f3qcm,
                &modes,
                &states,
                &coriolis,
            )
        };

        // this is worked on by resona and then enrgy so keep it out here
        let nstate = states.len();
        let mut eng = vec![0.0; nstate];

        if !self.rotor.is_sym_top() {
            resona(e0, &modes, &freq, &xcnst, &fermi1, &fermi2, &mut eng);
        } else {
            // straight from jan martin himself
            // println!(
            //     "resonance polyads for symmetric tops not yet implemented"
            // );
        }

        self.enrgy(&freq, &xcnst, &gcnst, &restst, &f3qcm, e0, &mut eng);

        // it's not obvious that the states are in this proper order, but by
        // construction that seems to be the case
        let mut corrs = Vec::new();
        let (n1dm, n2dm, n3dm) = Mode::count(&modes);
        for i in 1..n1dm + n2dm + n3dm + 1 {
            corrs.push(eng[i] - eng[0]);
        }

        // print_vib_states(&eng, &states);

        let quartic = Quartic::new(&self, &freq, &wila);
        let _sextic = Sextic::new(&self, &wila, &zmat, &freq, &f3qcm);
        let rots = self.rota(&rotnst, &states, &quartic);

        Output {
            harms,
            funds,
            rots,
            corrs,
        }
    }

    pub fn write(&self, filename: &str) -> Result<()> {
        use std::io::Write;
        let mut f = File::create(filename)?;
        writeln!(f, "{}", self)?;
        Ok(())
    }

    /// calculate the anharmonic constants and E_0 for an asymmetric top
    pub fn xcalc(
        &self,
        f4qcm: &F4qcm,
        freq: &Dvec,
        f3qcm: &F3qcm,
        zmat: &Tensor3,
        modes: &[Mode],
        fermi1: &[Fermi1],
        fermi2: &[Fermi2],
    ) -> (Dmat, f64) {
        let (ifrmchk, ifrm1) = self.make_fermi_checks(fermi1, fermi2);
        let mut xcnst = Dmat::zeros(self.nvib, self.nvib);
        // diagonal contributions to the anharmonic constants
        for k in 0..self.nvib {
            let kkkk = (k, k, k, k);
            let val1 = f4qcm[kkkk] / 16.0;
            let wk = freq[k].powi(2);
            let mut valu = 0.0;
            for l in 0..self.nvib {
                let val2 = f3qcm[(k, k, l)].powi(2);
                if ifrmchk[(k, k, l)] != 0 {
                    let val3 = 1.0 / (8.0 * freq[l]);
                    let val4 = 1.0 / (32.0 * (2.0 * freq[k] + freq[l]));
                    valu -= val2 * (val3 + val4);
                } else {
                    let wl = freq[l].powi(2);
                    let val3 = 8.0 * wk - 3.0 * wl;
                    let val4 = 16.0 * freq[l] * (4.0 * wk - wl);
                    valu -= val2 * val3 / val4;
                }
            }
            let value = val1 + valu;
            xcnst[(k, k)] = value;
        }
        // off-diagonal contributions to the anharmonic constants
        for k in 1..self.nvib {
            for l in 0..k {
                let kkll = (k, k, l, l);
                let val1 = f4qcm[kkll] / 4.0;
                let mut val2 = 0.0;
                for m in 0..self.nvib {
                    val2 -=
                        f3qcm[(k, k, m)] * f3qcm[(l, l, m)] / (4.0 * freq[m]);
                }

                let mut valu = 0.0;
                for m in 0..self.nvib {
                    let d1 = freq[k] + freq[l] + freq[m];
                    let d2 = freq[k] - freq[l] + freq[m];
                    let d3 = freq[k] + freq[l] - freq[m];
                    let d4 = -freq[k] + freq[l] + freq[m];
                    if ifrmchk[(l, m, k)] != 0 && m == l {
                        // case 1
                        let delta = 8.0 * (2.0 * freq[l] + freq[k]);
                        valu -= f3qcm[(k, l, m)].powi(2) / delta;
                    } else if ifrmchk[(k, m, l)] != 0 && k == m {
                        // case 2
                        let delta = 8.0 * (2.0 * freq[k] + freq[l]);
                        valu -= f3qcm[(k, l, m)].powi(2) / delta;
                    } else if ifrmchk[(k, l, m)] != 0 {
                        // case 3
                        let delta = 1.0 / d1 + 1.0 / d2 + 1.0 / d4;
                        valu -= f3qcm[(k, l, m)].powi(2) * delta / 8.0;
                    } else if ifrmchk[(l, m, k)] != 0 {
                        // case 4
                        let delta = 1.0 / d1 + 1.0 / d2 - 1.0 / d3;
                        valu -= f3qcm[(k, l, m)].powi(2) * delta / 8.0;
                    } else if ifrmchk[(k, m, l)] != 0 {
                        // case 5
                        let delta = 1.0 / d1 - 1.0 / d3 + 1.0 / d4;
                        valu -= f3qcm[(k, l, m)].powi(2) * delta / 8.0;
                    } else {
                        // default
                        let delta = -d1 * d2 * d3 * d4;
                        let val3 =
                            freq[m].powi(2) - freq[k].powi(2) - freq[l].powi(2);
                        valu -= 0.5 * f3qcm[(k, l, m)].powi(2) * freq[m] * val3
                            / delta;
                    }
                }
                let val5 = freq[k] / freq[l];
                let val6 = freq[l] / freq[k];
                let val7 = self.rotcon[0] * zmat[(k, l, 0)].powi(2)
                    + self.rotcon[1] * zmat[(k, l, 1)].powi(2)
                    + self.rotcon[2] * zmat[(k, l, 2)].powi(2);
                let val8 = (val5 + val6) * val7;
                let value = val1 + val2 + valu + val8;
                xcnst[(k, l)] = value;
                xcnst[(l, k)] = value;
            }
        }
        let e0 = make_e0(modes, f4qcm, f3qcm, freq, &ifrm1, &ifrmchk);
        (xcnst, e0)
    }

    /// calculate the anharmonic constants and E_0 for a symmetric top
    pub fn xcals(
        &self,
        f4qcm: &F4qcm,
        freq: &Dvec,
        f3qcm: &F3qcm,
        zmat: &Tensor3,
        fermi1: &[Fermi1],
        fermi2: &[Fermi2],
        modes: &[Mode],
        wila: &Dmat,
    ) -> (Dmat, Dmat, f64) {
        let (ia, ib) = (2, 1);
        let (_, n2dm, _) = Mode::count(modes);
        let (i1mode, i2mode, _) = Mode::partition(modes);
        // find out which of a(xz)tb or a(yz)tb are zero
        let (ixyz, _ia1, _ia2, _ix, _iy) = if !self.rotor.is_linear() {
            const TOL: f64 = 0.000001;
            let mut ixz = 0;
            let mut iyz = 0;
            for &(_, i2) in &i2mode {
                if wila[(i2, 3)].abs() <= TOL {
                    ixz += 1;
                }
                if wila[(i2, 4)].abs() <= TOL {
                    iyz += 1;
                }
            }
            if ixz > 0 && iyz > 0 {
                (1, 2, 3, 0, 1)
            } else if ixz > 0 {
                (1, 2, 3, 0, 1)
            } else if iyz > 0 {
                (0, 0, 4, 1, 0)
            } else {
                panic!("big problem in xcals");
            }
        } else {
            // actually everything I have here pretty much assumes the molecule
            // is not linear, so good to have this todo here
            todo!()
        };
        // NOTE skipping zeta checks, but they only print stuff

        let (ifrmchk, ifrm1) = self.make_fermi_checks(fermi1, fermi2);

        let e1 = make_e0(modes, f4qcm, f3qcm, freq, &ifrm1, &ifrmchk);
        let e2 = make_e2(modes, freq, f4qcm, f3qcm, &ifrm1);
        let e3 = make_e3(modes, freq, f3qcm, &ifrm1, &ifrmchk);
        let e0 = e1 + e2 + e3;

        // start calculating anharmonic constants
        let mut xcnst = Dmat::zeros(self.nvib, self.nvib);

        // nondeg-nondeg interactions
        for &k in &i1mode {
            let kkkk = (k, k, k, k);
            let val1 = f4qcm[kkkk] / 16.0;
            let wk = freq[k].powi(2);

            let mut valu = 0.0;
            for &l in &i1mode {
                let val2 = f3qcm[(k, k, l)].powi(2);
                if ifrm1.check(k, l) {
                    let val3 = 1.0 / (8.0 * freq[l]);
                    let val4 = 1.0 / (32.0 * (2.0 * freq[k] + freq[l]));
                    valu -= val2 * (val3 + val4);
                } else {
                    let wl = freq[l].powi(2);
                    let val3 = 8.0 * wk - 3.0 * wl;
                    let val4 = 16.0 * freq[l] * (4.0 * wk - wl);
                    valu -= val2 * val3 / val4;
                }
            }
            xcnst[(k, k)] = val1 + valu;
        }

        if self.rotor.is_diatomic() {
            todo!("goto funds section. return?")
        }

        for (kk, &k) in i1mode.iter().enumerate() {
            // might be kk-1 not sure
            for &l in i1mode.iter().take(kk) {
                let kkll = (k, k, l, l);
                let val1 = f4qcm[kkll] / 4.0;

                let mut val2 = 0.0;
                for &m in &i1mode {
                    val2 -=
                        f3qcm[(k, k, m)] * f3qcm[(l, l, m)] / (4.0 * freq[m]);
                }

                let mut valu = 0.0;
                for &m in &i1mode {
                    let klm = (k, m, l);
                    let d1 = freq[(k)] + freq[(l)] + freq[(m)];
                    let d2 = freq[(k)] - freq[(l)] + freq[(m)];
                    let d3 = freq[(k)] + freq[(l)] - freq[(m)];
                    let d4 = -freq[(k)] + freq[(l)] + freq[(m)];

                    if ifrmchk[(l, m, k)] != 0 {
                        let delta = 8.0 * (2.0 * freq[l] + freq[k]);
                        valu -= (f3qcm[klm].powi(2)) / delta;
                    } else if ifrmchk[(k, m, l)] != 0 {
                        let delta = 8.0 * (2.0 * freq[k] + freq[l]);
                        valu -= f3qcm[klm].powi(2) / delta;
                    } else if ifrmchk[(k, l, m)] != 0 {
                        let delta = 1.0 / d1 + 1.0 / d2 + 1.0 / d4;
                        valu -= f3qcm[klm].powi(2) * delta / 8.0;
                    } else if ifrmchk[(l, m, k)] != 0 {
                        let delta = 1.0 / d1 + 1.0 / d2 + 1.0 / d3;
                        valu -= f3qcm[klm].powi(2) * delta / 8.0;
                    } else if ifrmchk[(k, m, l)] != 0 {
                        let delta = 1.0 / d1 + 1.0 / d3 + 1.0 / d4;
                        valu -= f3qcm[klm].powi(2) * delta / 8.0;
                    } else {
                        let delta = -d1 * d2 * d3 * d4;
                        let val3 =
                            freq[m].powi(2) - freq[k].powi(2) - freq[l].powi(2);
                        valu -=
                            0.5 * (f3qcm[klm].powi(2)) * freq[m] * val3 / delta;
                    }
                }

                let val5 = freq[k] / freq[l];
                let val6 = freq[l] / freq[k];
                let val7 = self.rotcon[ia] * zmat[(k, l, 2)].powi(2);
                let val8 = (val5 + val6) * val7;
                let value = val1 + val2 + valu + val8;
                xcnst[(k, l)] = value;
                xcnst[(l, k)] = value;
            }
        }

        // nondeg-deg interactions
        for &k in &i1mode {
            for (ll, &(l, _)) in i2mode.iter().enumerate() {
                let kkll = (k, k, l, l);
                let val1 = f4qcm[kkll] / 4.0;

                let mut val2 = 0.0;
                for &m in &i1mode {
                    val2 -=
                        f3qcm[(k, k, m)] * f3qcm[(l, l, m)] / (4. * freq[m]);
                }

                let mut valu = 0.0;
                for &(m, _) in &i2mode {
                    // NOTE there's a ridiculous line in the fortran code that
                    // says
                    //
                    // if k == 7 || k == 12 { m = i2mode(mm, 2) }
                    //
                    // that can't be right in general and must have been a
                    // special case right?
                    let klm = (k, l, m);
                    let d1 = freq[(k)] + freq[(l)] + freq[(m)];
                    let d2 = freq[(k)] - freq[(l)] + freq[(m)];
                    let d3 = freq[(k)] + freq[(l)] - freq[(m)];
                    let d4 = -freq[(k)] + freq[(l)] + freq[(m)];

                    if ifrmchk[(l, m, k)] != 0 {
                        let delta = 8.0 * (2.0 * freq[(l)] + freq[(k)]);
                        valu -= (f3qcm[(klm)].powi(2)) / delta;
                    } else if ifrmchk[(k, l, m)] != 0 {
                        let delta = 1.0 / d1 + 1.0 / d2 + 1.0 / d4;
                        valu = valu - (f3qcm[(klm)].powi(2)) * delta / 8.0;
                    } else if ifrmchk[(l, m, k)] != 0 {
                        let delta = 1.0 / d1 + 1.0 / d2 + 1.0 / d3;
                        valu = valu - (f3qcm[(klm)].powi(2)) * delta / 8.0;
                    } else if ifrmchk[(k, m, l)] != 0 {
                        let delta = 1.0 / d1 + 1.0 / d3 + 1.0 / d4;
                        valu = valu - (f3qcm[(klm)].powi(2)) * delta / 8.0;
                    } else {
                        let delta = -d1 * d2 * d3 * d4;
                        let val3 = freq[(m)].powi(2)
                            - freq[(k)].powi(2)
                            - freq[(l)].powi(2);
                        valu -= 0.5 * (f3qcm[(klm)].powi(2)) * freq[(m)] * val3
                            / delta;
                    }
                }
                let val5 = freq[(k)] / freq[(l)];
                let val6 = freq[(l)] / freq[(k)];
                let val7 = self.rotcon[(ib)]
                    * (zmat[(k, l, 0)].powi(2) + zmat[(k, l, 1)].powi(2));
                let val8 = (val5 + val6) * val7;
                let value = val1 + val2 + valu + val8;
                let l2 = i2mode[ll].1;
                // 3,2 and 4,2 are wrong, as are 2,3 and 2,4
                xcnst[(k, l)] = value;
                xcnst[(l, k)] = value;
                xcnst[(k, l2)] = value;
                xcnst[(l2, k)] = value;
            }
        }

        // deg-deg modes
        for (kk, &(k, _)) in i2mode.iter().enumerate() {
            let val1 = f4qcm[(k, k, k, k)] / 16.0;
            let wk = freq[k].powi(2);

            let mut valu = 0.0;
            for &l in &i1mode {
                let val2 = f3qcm[(k, k, l)].powi(2);
                if ifrm1.check(k, l) {
                    let val3 = 1.0 / (8.0 * freq[(l)]);
                    let val4 = 1.0 / (32.0 * (2.0 * freq[(k)] + freq[(l)]));
                    valu -= val2 * (val3 + val4);
                } else {
                    let wl = freq[(l)] * freq[(l)];
                    let val3 = 8.0 * wk - 3.0 * wl;
                    let val4 = 16.0 * freq[(l)] * (4.0 * wk - wl);
                    valu -= val2 * val3 / val4;
                }
            }

            let mut valus = 0.0;
            for &(l, _) in &i2mode {
                let val2 = f3qcm[(k, k, l)].powi(2);
                if ifrm1.check(k, l) {
                    let val3 = 1.0 / (8.0 * freq[(l)]);
                    let val4 = 1.0 / (32.0 * (2.0 * freq[(k)] + freq[(l)]));
                    valus -= val2 * (val3 + val4);
                } else {
                    let wl = freq[(l)] * freq[(l)];
                    let val3 = 8.0 * wk - 3.0 * wl;
                    let val4 = 16.0 * freq[(l)] * (4.0 * wk - wl);
                    valus -= val2 * val3 / val4;
                }
            }

            let value = val1 + valu + valus;
            let k2 = i2mode[kk].1;
            xcnst[(k, k)] = value;
            xcnst[(k2, k2)] = value;
        }
        for kk in 1..n2dm {
            let k = i2mode[kk].0;
            // might be -1
            for ll in 0..kk {
                let (l, l2) = i2mode[ll];
                let val1 = (f4qcm[(k, k, l, l)] + f4qcm[(k, k, l2, l2)]) / 8.0;

                let val2: f64 = i1mode
                    .iter()
                    .map(|&m| {
                        -(f3qcm[(k, k, m)] * f3qcm[(l, l, m)] / (4.0 * freq[m]))
                    })
                    .sum();

                let valu: f64 = i1mode
                    .iter()
                    .map(|&m| {
                        let d1 = freq[(k)] + freq[(l)] + freq[(m)];
                        let d2 = freq[(k)] - freq[(l)] + freq[(m)];
                        let d3 = freq[(k)] + freq[(l)] - freq[(m)];
                        let d4 = -freq[(k)] + freq[(l)] + freq[(m)];

                        if ifrmchk[(k, l, m)] != 0 {
                            let delta = 1.0 / d1 + 1.0 / d2 + 1.0 / d4;
                            -(f3qcm[(k, l, m)].powi(2)) * delta / 16.0
                        } else if ifrmchk[(l, m, k)] != 0 {
                            let delta = 1.0 / d1 + 1.0 / d2 + 1.0 / d3;
                            -(f3qcm[(k, l, m)].powi(2)) * delta / 16.0
                        } else if ifrmchk[(k, m, l)] != 0 {
                            let delta = 1.0 / d1 + 1.0 / d3 + 1.0 / d4;
                            -(f3qcm[(k, l, m)].powi(2)) * delta / 16.0
                        } else {
                            let delta = -d1 * d2 * d3 * d4;
                            let val3 = freq[(m)].powi(2)
                                - freq[(k)].powi(2)
                                - freq[(l)].powi(2);
                            -0.25
                                * (f3qcm[(k, l, m)].powi(2))
                                * freq[(m)]
                                * val3
                                / delta
                        }
                    })
                    .sum();

                let valus: f64 = i2mode
                    .iter()
                    .map(|&(m, _)| {
                        let d1 = freq[(k)] + freq[(l)] + freq[(m)];
                        let d2 = freq[(k)] - freq[(l)] + freq[(m)];
                        let d3 = freq[(k)] + freq[(l)] - freq[(m)];
                        let d4 = -freq[(k)] + freq[(l)] + freq[(m)];

                        let klm = (k, l, m);
                        if ifrmchk[(l, m, k)] != 0 {
                            let delta = 8.0 * (2.0 * freq[(l)] + freq[(k)]);
                            -(f3qcm[(klm)].powi(2)) / delta
                        } else if ifrmchk[(k, m, l)] != 0 {
                            let delta = 8.0 * (2.0 * freq[(k)] + freq[(l)]);
                            -(f3qcm[(klm)].powi(2)) / delta
                        } else if ifrmchk[(k, l, m)] != 0 {
                            let delta = 1.0 / d1 + 1.0 / d2 + 1.0 / d4;
                            -(f3qcm[(klm)].powi(2)) * delta / 8.0
                        } else if ifrmchk[(l, m, k)] != 0 {
                            let delta = 1.0 / d1 + 1.0 / d2 + 1.0 / d3;
                            -(f3qcm[(klm)].powi(2)) * delta / 8.0
                        } else if ifrmchk[(k, m, l)] != 0 {
                            let delta = 1.0 / d1 + 1.0 / d3 + 1.0 / d4;
                            -(f3qcm[(klm)].powi(2)) * delta / 8.0
                        } else {
                            let delta = -d1 * d2 * d3 * d4;
                            let val3 = freq[(m)].powi(2)
                                - freq[(k)].powi(2)
                                - freq[(l)].powi(2);
                            -0.5 * (f3qcm[(klm)].powi(2)) * freq[(m)] * val3
                                / delta
                        }
                    })
                    .sum();

                let val5 = freq[(k)] / freq[(l)];
                let val6 = freq[(l)] / freq[(k)];
                let val7 = 0.5 * self.rotcon[(ia)] * (zmat[(k, l2, 2)].powi(2))
                    + self.rotcon[(ib)] * (zmat[(k, l, ixyz)].powi(2));
                let val8 = (val5 + val6) * val7;
                let value = val1 + val2 + valu + valus + val8;
                let k2 = i2mode[kk].1;
                xcnst[(k, l)] = value;
                xcnst[(l, k)] = value;
                xcnst[(k2, l2)] = value;
                xcnst[(l2, k2)] = value;
            }
        }

        // g constants for degenerate modes
        let mut gcnst = Dmat::zeros(self.nvib, self.nvib);
        for kk in 0..n2dm {
            let (k, k2) = i2mode[kk];
            let val1 = -f4qcm[(k, k, k, k)] / 48.0;
            let wk = freq[k].powi(2);

            let valu: f64 = i1mode
                .iter()
                .map(|&l| {
                    let val2 = f3qcm[(k, k, l)].powi(2);
                    if ifrm1.check(k, l) {
                        let val3 = 32.0 * (2.0 * freq[(k)] + freq[(l)]);
                        val2 / val3
                    } else {
                        let wl = freq[(l)] * freq[(l)];
                        let val3 = val2 * freq[(l)];
                        let val4 = 16.0 * (4.0 * wk - wl);
                        -val3 / val4
                    }
                })
                .sum();

            let valus: f64 = i2mode
                .iter()
                .map(|&(l, _)| {
                    let val2 = f3qcm[(k, k, l)].powi(2);
                    if ifrm1.check(k, l) {
                        let val3 = 1.0 / (8.0 * freq[(l)]);
                        let val4 = 1.0 / (32.0 * (2.0 * freq[(k)] + freq[(l)]));
                        -val2 * (val3 + val4)
                    } else {
                        let wl = freq[(l)] * freq[(l)];
                        let val3 = 8.0 * wk - wl;
                        let val4 = 16.0 * freq[(l)] * (4.0 * wk - wl);
                        val2 * val3 / val4
                    }
                })
                .sum();

            let val7 = self.rotcon[(ia)] * (zmat[(k, k2, 2)].powi(2));
            let value = val1 + valu + valus + val7;
            gcnst[(k, k)] = value;
            gcnst[(k2, k2)] = value;
        }
        for kk in 1..n2dm {
            let (k, k2) = i2mode[kk];
            for ll in 0..kk {
                let (l, l222) = i2mode[ll];

                let valu: f64 = i1mode
                    .iter()
                    .map(|&m| {
                        let d1 = freq[(k)] + freq[(l)] + freq[(m)];
                        let d2 = freq[(k)] - freq[(l)] + freq[(m)];
                        let d3 = freq[(k)] + freq[(l)] - freq[(m)];
                        let d4 = -freq[(k)] + freq[(l)] + freq[(m)];

                        let klm = (k, l, m);
                        if ifrmchk[(k, l, m)] != 0 {
                            let delta = 1.0 / d1 + 1.0 / d2 + 1.0 / d4;
                            -(f3qcm[(klm)].powi(2)) * delta / 8.0
                        } else if ifrmchk[(l, m, k)] != 0 {
                            let delta = 1.0 / d1 + 1.0 / d2 + 1.0 / d3;
                            -(f3qcm[(klm)].powi(2)) * delta / 8.0
                        } else if ifrmchk[(k, m, l)] != 0 {
                            let delta = 1.0 / d1 + 1.0 / d3 + 1.0 / d4;
                            -(f3qcm[(klm)].powi(2)) * delta / 8.0
                        } else {
                            let delta = -d1 * d2 * d3 * d4;
                            let val3 = freq[(m)] * freq[(k)] * freq[(l)];
                            0.5 * (f3qcm[(klm)].powi(2)) * val3 / delta
                        }
                    })
                    .sum();

                let valus: f64 = i2mode
                    .iter()
                    .map(|&(m, _)| {
                        let d1 = freq[(k)] + freq[(l)] + freq[(m)];
                        let d2 = freq[(k)] - freq[(l)] + freq[(m)];
                        let d3 = freq[(k)] + freq[(l)] - freq[(m)];
                        let d4 = -freq[(k)] + freq[(l)] + freq[(m)];

                        let klm = (k, l, m);
                        if ifrmchk[(l, m, k)] != 0 {
                            let delta = 8.0 * (2.0 * freq[(l)] + freq[(k)]);
                            -(f3qcm[(klm)].powi(2)) / delta
                        } else if ifrmchk[(k, m, l)] != 0 {
                            let delta = 8.0 * (2.0 * freq[(k)] + freq[(l)]);
                            -(f3qcm[(klm)].powi(2)) / delta
                        } else if ifrmchk[(k, l, m)] != 0 {
                            let delta = 1.0 / d1 + 1.0 / d2 + 1.0 / d4;
                            -(f3qcm[(klm)].powi(2)) * delta / 8.0
                        } else if ifrmchk[(l, m, k)] != 0 {
                            let delta = 1.0 / d1 + 1.0 / d2 + 1.0 / d3;
                            -(f3qcm[(klm)].powi(2)) * delta / 8.0
                        } else if ifrmchk[(k, m, l)] != 0 {
                            let delta = 1.0 / d1 + 1.0 / d3 + 1.0 / d4;
                            -(f3qcm[(klm)].powi(2)) * delta / 8.0
                        } else {
                            let delta = -d1 * d2 * d3 * d4;
                            let val3 = freq[(m)] * freq[(k)] * freq[(l)];
                            -(f3qcm[(klm)].powi(2)) * val3 / delta
                        }
                    })
                    .sum();

                let val7 =
                    -2.0 * self.rotcon[(ib)] * (zmat[(k, l, ixyz)].powi(2))
                        + self.rotcon[(ia)] * (zmat[(k, l222, 2)].powi(2))
                        + 2.0
                            * self.rotcon[(ia)]
                            * zmat[(k, k2, 2)]
                            * zmat[(l, l222, 2)];
                // NOTE last two zmat values are the same sign instead of
                // opposite for nh3, causes an issue in gcnst. hopefully it will
                // cause small issues later on.. geometry looks okay, just
                // pointing the opposite direction on the z axis from the
                // fortran version. an element with the other sign exists in my
                // zmat, but I don't really want to start randomly changing
                // stuff without more tests. If I see multiple cases like this,
                // I will feel more comfortable changing the index in the code
                let value = valu + valus + val7;
                gcnst[(k, l)] = value;
                gcnst[(l, k)] = value;
                gcnst[(k2, l222)] = value;
                gcnst[(l222, k2)] = value;
            }
        }
        (xcnst, gcnst, e0)
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
}

/// make the second component of E0 for symmetric tops
fn make_e2(
    modes: &[Mode],
    freq: &Dvec,
    f4qcm: &F4qcm,
    f3qcm: &F3qcm,
    ifrm1: &Ifrm1,
) -> f64 {
    let (n1dm, n2dm, _) = Mode::count(modes);
    let (i1mode, i2mode, _) = Mode::partition(modes);
    // e2
    // sss and ssss terms
    let mut f4s = 0.0;
    let mut f3s = 0.0;
    let mut f3kss = 0.0;
    for kk in 0..n2dm {
        let (k, _) = i2mode[kk];
        let wk = freq[k].powi(2);
        f4s += f4qcm[(k, k, k, k)] / 48.0;
        f3s += 11.0 * f3qcm[(k, k, k)].powi(2) / (freq[k] * 144.0);

        // kss and fss terms
        for ll in 0..n1dm {
            let l = i1mode[ll];
            let zval4 = f3qcm[(k, k, l)].powi(2);
            let wl = freq[l].powi(2);
            if ifrm1.check(k, l) {
                let delta4 = 2.0 * (2.0 * freq[k] + freq[l]);
                f3kss += zval4 / (16.0 * delta4);
            } else {
                let delta4 = 4.0 * wk - wl;
                f3kss += zval4 * freq[l] / (16.0 * delta4);
            }
        }
    }
    f4s + f3s + f3kss
}

/// make the third component of E0 for symmetric tops
fn make_e3(
    modes: &[Mode],
    freq: &Dvec,
    f3qcm: &F3qcm,
    ifrm1: &Ifrm1,
    ifrmchk: &tensor::Tensor3<usize>,
) -> f64 {
    let (n1dm, n2dm, _) = Mode::count(modes);
    let (i1mode, i2mode, _) = Mode::partition(modes);
    let mut f3kst = 0.0;
    let mut f3sst = 0.0;
    let mut f3stu = 0.0;
    // they check that ilin == 0 again here, but this all should only be called
    // if the molecule isn't linear
    for kk in 0..n2dm {
        let (k, _) = i2mode[kk];
        let wk = freq[k].powi(2);
        for ll in 0..n2dm {
            let (l, _) = i2mode[ll];
            if k == l {
                continue;
            }
            let wl = freq[l].powi(2);
            let zval5 = f3qcm[(k, k, l)].powi(2);
            if ifrm1.check(k, l) {
                // very encouraging comment from jan martin here: "this is
                // completely hosed!!!! dimension is cm-1 and should be cm !!!
                // perhaps he's multiplying by 2*FREQ(L) when he should divide?"
                // I'm assuming he fixed this and the code I've translated is
                // right.
                let delta5 =
                    (8.0 * wk + wl) / (2.0 * freq[k] + freq[l]) / (2.0 * wl);
                f3sst += zval5 * delta5 / 16.0;
            } else {
                let delta5 = (8.0 * wk + wl) / (freq[l] * (4.0 * wk - wl));
                f3sst += zval5 * delta5 / 16.0;
            }

            for mm in 0..n1dm {
                let m = i1mode[mm];
                if k <= l {
                    f3kst = 0.0;
                } else {
                    let zval6 = f3qcm[(k, l, m)].powi(2);
                    let xkst = freq[(k)] * freq[(l)] * freq[(m)];
                    let d1 = freq[(k)] + freq[(l)] + freq[(m)];
                    let d2 = freq[(k)] - freq[(l)] + freq[(m)];
                    let d3 = freq[(k)] + freq[(l)] - freq[(m)];
                    let d4 = freq[(k)] - freq[(l)] - freq[(m)];
                    if ifrmchk[(k, l, m)] != 0 {
                        let delta6 = 1.0 / d1 + 1.0 / d2 + 1.0 / d4;
                        f3kst -= zval6 * delta6 / 8.0;
                    } else if ifrmchk[(l, m, k)] != 0 {
                        let delta6 = 1.0 / d1 + 1.0 / d2 - 1.0 / d3;
                        f3kst -= zval6 * delta6 / 8.0;
                    } else if ifrmchk[(k, m, l)] != 0 {
                        let delta6 = 1.0 / d1 - 1.0 / d3 + 1.0 / d4;
                        f3kst -= zval6 * delta6 / 8.0;
                    } else {
                        let delta6 = d1 * d2 * d3 * d4;
                        f3kst -= zval6 * xkst / (2.0 * delta6);
                    }
                }
            }

            // stu term
            for mm in 0..n2dm {
                let (m, _) = i2mode[mm];
                if k <= l || l <= m {
                    f3stu = 0.0;
                } else {
                    let zval7 = f3qcm[(k, l, m)].powi(2);
                    let xstu = freq[(k)] * freq[(l)] * freq[(m)];
                    let d1 = freq[(k)] + freq[(l)] + freq[(m)];
                    let d2 = freq[(k)] - freq[(l)] + freq[(m)];
                    let d3 = freq[(k)] + freq[(l)] - freq[(m)];
                    let d4 = freq[(k)] - freq[(l)] - freq[(m)];
                    if ifrmchk[(k, l, m)] != 0 {
                        let delta7 = 1.0 / d1 + 1.0 / d2 + 1.0 / d4;
                        f3stu -= 2.0 * zval7 * delta7 / 4.0;
                    } else if ifrmchk[(l, m, k)] != 0 {
                        let delta7 = 1.0 / d1 + 1.0 / d2 - 1.0 / d3;
                        f3stu -= 2.0 * zval7 * delta7 / 4.0;
                    } else if ifrmchk[(k, m, l)] != 0 {
                        let delta7 = 1.0 / d1 - 1.0 / d3 + 1.0 / d4;
                        f3stu -= 2.0 * zval7 * delta7 / 4.0;
                    } else {
                        let delta7 = d1 * d2 * d3 * d4;
                        f3stu -= 2.0 * zval7 * xstu / (delta7);
                    }
                }
            }
        }
    }
    f3kst + f3sst + f3stu
}

fn make_sym_funds(
    modes: &Vec<Mode>,
    freq: &Dvec,
    xcnst: &Dmat,
    gcnst: &Option<Dmat>,
) -> (Vec<f64>, Vec<f64>) {
    let (n1dm, n2dm, _) = Mode::count(modes);
    let (i1mode, i2mode, _) = Mode::partition(modes);
    let mut harms = Vec::new();
    let mut funds = Vec::new();
    for ii in 0..n1dm {
        let i = i1mode[ii];
        let mut val = freq[i] + xcnst[(i, i)] * 2.0;
        for jj in 0..n1dm {
            let j = i1mode[jj];
            if j != i {
                val += 0.5 * xcnst[(i, j)];
            }
        }
        for jj in 0..n2dm {
            let j = i2mode[jj].0;
            val += xcnst[(i, j)];
        }
        harms.push(freq[i]);
        funds.push(val);
    }
    for ii in 0..n2dm {
        let i = i2mode[ii].0;
        let mut val =
            freq[i] + 3.0 * xcnst[(i, i)] + gcnst.as_ref().unwrap()[(i, i)];
        for jj in 0..n1dm {
            let j = i1mode[jj];
            val += 0.5 * xcnst[(i, j)];
        }
        for jj in 0..n2dm {
            let j = i2mode[jj].0;
            if j != i {
                val += xcnst[(i, j)]
            }
        }
        harms.push(freq[i]);
        funds.push(val);
    }
    (harms, funds)
}

/// Builds a HashMap of Coriolis resonances to their corresponding axes. NOTE
/// like the fermi resonances, this overwrites earlier resonances. should it use
/// them all?
fn make_icorol(coriolis: &[Coriolis]) -> HashMap<(usize, usize), usize> {
    let mut icorol = HashMap::new();
    for &Coriolis { i, j, axis } in coriolis {
        icorol.insert((i, j), axis as usize);
        icorol.insert((j, i), axis as usize);
    }
    icorol
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
    pub harms: Vec<f64>,

    /// partially resonance-corrected anharmonic frequencies
    pub funds: Vec<f64>,

    /// fully resonance-corrected anharmonic frequencies
    pub corrs: Vec<f64>,

    /// vibrationally averaged rotational constants
    pub rots: Vec<Rot>,
}

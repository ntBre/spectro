#![feature(test, stmt_expr_attributes)]

use std::{
    cmp::{max, min},
    collections::{HashMap, HashSet},
    f64::consts::PI,
    fmt::Debug,
    fs::File,
    io::Result,
};

use consts::ALPHA_CONST;
use dummy::Dummy;
use f3qcm::F3qcm;
use f4qcm::F4qcm;
use ifrm1::Ifrm1;
use ifrm2::Ifrm2;
use nalgebra::DMatrix;
use quartic::Quartic;
use resonance::{Coriolis, Fermi1, Fermi2, Restst};
use rot::Rot;
use rotor::{Rotor, ROTOR_EPS};
use state::State;
use symm::{Axis, Molecule};
use tensor::Tensor4;
use utils::*;

mod alphas;
mod consts;
mod dummy;
mod f3qcm;
mod f4qcm;
mod ifrm1;
mod ifrm2;
mod load;
mod mode;
mod quartic;
mod resonance;
mod rot;
mod rotor;
mod run;
mod sextic;
mod state;
mod utils;
mod xcals;

pub use mode::*;

#[cfg(test)]
mod tests;

type Tensor3 = tensor::tensor3::Tensor3<f64>;
type Mat3 = nalgebra::Matrix3<f64>;
type Dvec = nalgebra::DVector<f64>;
type Dmat = nalgebra::DMatrix<f64>;

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

    /// index of the atom aligned with the X axis. symmetric tops only
    pub iatom: usize,

    /// order of the highest rotation axis. symmetric tops only
    pub axis_order: usize,

    /// the actual Axis with order `axis_order`. symmetric tops only
    pub axis: Axis,
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
                let bva = self.rotcon[axis] + suma;
                rotnst[(ist, axis)] = bva;
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

        let (_, ifrm1, ifrm2) = self.make_fermi_checks(fermi1, fermi2);

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

    /// Note that `ifrm1` and `ifrm2` are deduplicated because of the Hash, but
    /// `ifrmchk` includes all of the resonances. This is the desired behavior
    /// from the Fortran version. I think this deduplication is a mistake, but
    /// I'm reproducing the Fortran behavior for now. It could be an intentional
    /// decision to prevent double-counting, but the way it's implemented by
    /// overwriting the array index when you read another resonance makes it
    /// look like a mistake. Why would you ever want the second resonance input
    /// to take precedence if you were doing this intentionally?
    fn make_fermi_checks(
        &self,
        fermi1: &[Fermi1],
        fermi2: &[Fermi2],
    ) -> (tensor::Tensor3<usize>, Ifrm1, Ifrm2) {
        let mut ifrmchk = tensor::tensor3::Tensor3::<usize>::zeros(
            self.nvib, self.nvib, self.nvib,
        );
        // using a hash here instead of an array because I need some way to
        // signal that the value is not there. in fortran they use an array of
        // zeros because zero will never be a valid index. I could use -1, but
        // then the vec has to be of isize and I have to do a lot of casting.
        let mut ifrm1 = Ifrm1::new();
        let mut ifrm2 = Ifrm2::new();
        for f in fermi1 {
            ifrmchk[(f.i, f.i, f.j)] = 1;
            ifrm1.insert(f.i, f.j);
            ifrm2.insert((f.i, f.i), f.j);
        }
        for f in fermi2 {
            ifrmchk[(f.i, f.j, f.k)] = 1;
            ifrmchk[(f.j, f.i, f.k)] = 1;
            ifrm2.insert((f.i, f.j), f.k);
        }
        (ifrmchk, ifrm1, ifrm2)
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
    pub fn rot3rd(&self, f3x: Tensor3) -> Tensor3 {
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
                    let temp2 = self.axes.transpose() * a * self.axes;
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
                        val[ii] = ret[(j, k, ia)] * self.axes[(0, ii)]
                            + ret[(j, k, ia + 1)] * self.axes[(1, ii)]
                            + ret[(j, k, ia + 2)] * self.axes[(2, ii)];
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
    pub fn rot4th(&self, f4x: Tensor4) -> Tensor4 {
        let egt = self.axes.transpose();
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
                        let temp2 = egt * a * self.axes;
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
                        let temp2 = egt * a * self.axes;
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
        let irep = 0;
        let (ic, _) = princ_cart(irep);
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
                State::I1st(_) => {
                    ret.push(Rot::new(states[nst].clone(), bza, bxa, bya));
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
    fn rotor_type(&self, moms: &nalgebra::Vector3<f64>) -> Rotor {
        if self.geom.atoms.len() == 2 {
            return Rotor::Diatomic;
        }
        let close = |a, b| f64::abs(a - b) < ROTOR_EPS;
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

    /// automatic alignment of the degenerate modes of symmetric tops, according
    /// to the convention given at the beginning of Papousek and Aliev. most
    /// vibrations are aligned so that the first component is symmetric under a
    /// mirror plane through atom number iatom (and the second component is
    /// antisymmetric) however, for planar molecules the out-of-plane degenerate
    /// modes are aligned using the c2 rotation instead of the mirror plane.
    /// also aligns linear molecule modes, assuming molecule is already aligned
    /// on the z axis. for linear molecules, IATOM will obviously be on the z
    /// axis.
    fn bdegnl(&self, freq: &Dvec, lxm: &mut Dmat, w: &Vec<f64>, lx: &mut Dmat) {
        /// cutoff for considering two modes degenerate
        const TOL: f64 = 0.05;
        for ii in 0..self.nvib - 1 {
            for jj in ii + 1..self.nvib {
                if (freq[ii] - freq[jj]).abs() < TOL {
                    /// size of the expected errors in lxm
                    const TOLER: f64 = 1e-5;

                    let ncomp1 = 3 * self.iatom;
                    let imode1 = ii;
                    let imode2 = jj;
                    let mut izero = 0;
                    let mut iz1x = 0;
                    let mut iz2x = 0;
                    let mut iz2y = 0;
                    if lxm[(ncomp1, imode1)].abs() < TOLER {
                        izero += 1;
                        iz1x = 1;
                    }
                    if lxm[(ncomp1, imode2)].abs() < TOLER {
                        izero += 1;
                        iz2x = 1;
                    }
                    let ncomp2 = ncomp1 + 1;
                    if lxm[(ncomp2, imode1)].abs() < TOLER {
                        izero += 1;
                    }
                    if lxm[(ncomp2, imode2)].abs() < TOLER {
                        iz2y = 1;
                        izero += 1;
                    }

                    // deg modes of interest have components in the xy plane
                    let theta = if izero != 4 {
                        if iz1x == 0 {
                            f64::atan(
                                lxm[(ncomp1, imode2)] / lxm[(ncomp1, imode1)],
                            )
                        } else {
                            if iz2x == 0 {
                                0.5 * PI
                            } else {
                                if iz2y == 0 {
                                    f64::atan(
                                        -1.0 * lxm[(ncomp2, imode1)]
                                            / lxm[(ncomp1, imode2)],
                                    )
                                } else {
                                    0.5 * PI
                                }
                            }
                        }
                    } else {
                        if self.rotor.is_linear() {
                            todo!("says increment iatom and go back to start");
                        }
                        let ncomp3 = ncomp1 + 2;
                        if lxm[(ncomp3, imode2)].abs() > TOLER {
                            let theta = f64::atan(
                                -1.0 * lxm[(ncomp3, imode1)]
                                    / lxm[(ncomp3, imode2)],
                            );
                            theta
                        } else {
                            if lxm[(ncomp3, imode1)] > TOLER {
                                let theta = 0.5 * PI;
                                theta
                            } else {
                                panic!("cannot determine mode alignment")
                            }
                        }
                    };

                    let c = theta.cos();
                    let s = theta.sin();
                    for i in 0..self.n3n {
                        let temp = c * lxm[(i, imode1)] + s * lxm[(i, imode2)];
                        lxm[(i, imode2)] =
                            c * lxm[(i, imode2)] - s * lxm[(i, imode1)];
                        lxm[(i, imode1)] = temp;
                    }

                    // so now the modes are in the xz and yz planes
                    // respectively now we need to check their relative
                    // signs, so that they will transform properly (i think
                    // this means have positive angular momentum for l=+1
                    // wavefunctions???) first do linear molecules:

                    if self.rotor.is_linear() {
                        todo!("bdegnl.f:136");
                    }

                    // usize so already abs
                    let nabs = self.axis_order as f64;

                    // the minus sign is Papousek's definition of the
                    // symmetry operation Cn (??!)
                    let other = self.geom.rotate(-360.0 / nabs, &self.axis);
                    let buddies = self.geom.detect_buddies(&other, 1e-6);

                    // might be the wrong axis or wrong point group
                    let iatom2 = buddies[self.iatom].expect(&format!(
                        "can't find symmetry related atom to atom {}",
                        self.iatom
                    ));

                    let alpha = (-2.0 * PI) / (nabs);

                    let mut s = 0.0;
                    let ncomp21 = 3 * iatom2;
                    let ncomp22 = 3 * iatom2 + 1;
                    let ncomp23 = 3 * iatom2 + 2;
                    let mut iflag = false;
                    for is in 0..5 {
                        let is2 = (is + is) as f64;
                        s += 1.0;
                        if is2 >= nabs {
                            continue;
                        }
                        if iflag {
                            continue;
                        }
                        if izero != 4 {
                            let (test, test2, test3) =
                                if f64::abs(f64::cos(alpha)) > TOLER {
                                    let test = f64::cos(alpha)
                                        * lxm[(ncomp1, imode1)]
                                        - f64::cos(alpha * s)
                                            * lxm[(ncomp21, imode1)];
                                    let test2 = lxm[(ncomp21, imode2)]
                                        * f64::sin(-alpha * s);
                                    let test3 = (f64::abs(test)
                                        - f64::abs(test2))
                                        / lxm[(ncomp1, imode1)];
                                    (test, test2, test3)
                                } else {
                                    let test = f64::sin(alpha)
                                        * lxm[(ncomp1, imode1)]
                                        - f64::cos(alpha * s)
                                            * lxm[(ncomp22, imode1)];
                                    let test2 = lxm[(ncomp22, imode2)]
                                        * f64::sin(-alpha * s);
                                    let test3 = (f64::abs(test)
                                        - f64::abs(test2))
                                        / lxm[(ncomp1, imode1)];
                                    (test, test2, test3)
                                };
                            if f64::abs(test3) < 0.001 {
                                iflag = true;
                                if test * test2 < 0.0 {
                                    for ii in 0..self.n3n {
                                        lxm[(ii, imode2)] =
                                            -1.0 * lxm[(ii, imode2)];
                                    }
                                }
                            }
                        } else {
                            // assume it's the same as the other definition
                            let ncomp3 = ncomp1 + 2;
                            let mut test = lxm[(ncomp3, imode2)];
                            // TODO improper rotations negate test here

                            test -=
                                f64::cos(alpha * s) * lxm[(ncomp23, imode2)];
                            let test2 = -1.0
                                * lxm[(ncomp23, imode1)]
                                * f64::sin(-alpha * s);
                            let test3 = (test.abs() - test2.abs())
                                / lxm[(ncomp3, imode2)];
                            if test3.abs() < 0.001 {
                                if test * test2 < 0.0 {
                                    for ii in 0..self.n3n {
                                        lxm[(ii, imode2)] =
                                            -1.0 * lxm[(ii, imode2)];
                                    }
                                }
                            }
                        }
                    }

                    for ij in 0..2 {
                        let mut ijj = imode1;
                        if ij == 1 {
                            ijj = imode2;
                        }
                        for ia in 0..self.natom {
                            let vmass = 1.0 / w[ia].sqrt();
                            for ix in 0..3 {
                                let ii = 3 * ia + ix;
                                lx[(ii, ijj)] = lxm[(ii, ijj)] * vmass;
                            }
                        }
                    }
                }
            }
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
        let (ifrmchk, ifrm1, _) = self.make_fermi_checks(fermi1, fermi2);
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

    /// Calculate the zeta matrices and Wilson A matrix. Zeta is for the
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

    /// compute the rotational energy levels of a symmetric top
    fn rots(
        &self,
        rotnst: &Dmat,
        states: &[State],
        quartic: &Quartic,
    ) -> Vec<Rot> {
        let (ia, ib) = if self.rotor.is_prolate() {
            (0, 1)
        } else {
            (2, 1)
        };
        let irep = 5;
        let (ic, _) = princ_cart(irep);
        let (nstop, _) = rotnst.shape();
        let (b1s, b2s, b3s) = quartic.srots();
        let mut ret = Vec::new();
        // TODO want only fundamental states
        for nst in 0..nstop {
            match &states[nst] {
                State::I1st(v) => {
                    // accept all zeros=ground state or a single 1=fund
                    if v.iter().sum::<usize>() > 1 {
                        continue;
                    }
                }
                State::I2st(v) => {
                    if v.iter().filter(|&&p| p == (1, 1)).count() != 1 {
                        continue;
                    }
                }
                State::I3st(_) => todo!(),
                State::I12st { i1st: _, i2st: _ } => continue,
            }
            let vib1 = rotnst[(nst, ia)] - self.rotcon[ia];
            let vib2 = rotnst[(nst, ib)] - self.rotcon[ib];

            let mut vibr = [0.0; 3];
            vibr[ic[0]] = vib1;
            vibr[ic[1]] = vib2;
            vibr[ic[2]] = vib2;
            let (bxs, bys, bzs) = if self.rotor.is_prolate() {
                let bxs = b1s + vibr[(1)];
                let bys = b2s + vibr[(0)];
                let bzs = b3s + vibr[(2)];
                (bxs, bys, bzs)
            } else {
                // only difference is order of vibr indices here
                let bxs = b1s + vibr[(2)];
                let bys = b2s + vibr[(0)];
                let bzs = b3s + vibr[(1)];
                (bxs, bys, bzs)
            };
            match &states[nst] {
                State::I1st(_) | State::I2st(_) => {
                    ret.push(Rot::new(states[nst].clone(), bys, bxs, bzs));
                }
                _ => (),
            }
        }
        ret
    }
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

use serde::{Deserialize, Serialize};
use symm::Irrep;

use super::{make_sym_funds, resona, Output, Spectro};
use crate::sextic::Sextic;
use crate::utils::{
    force3, force4, linalg::symm_eigen_decomp, make_funds, to_wavenumbers,
};
use crate::{consts::FACT2, quartic::Quartic, resonance::Restst};
use crate::{
    load_fc2, load_fc3, load_fc4, Derivative, Dmat, Dvec, F3qcm, F4qcm, Mode,
};
use std::error::Error;
use std::path::Path;

#[derive(Serialize, Deserialize)]
pub struct SpectroFinish {
    pub spectro: Spectro,
    pub freq: Dvec,
    pub f3qcm: F3qcm,
    pub f4qcm: F4qcm,
    pub irreps: Vec<Irrep>,
    pub lxm: Dmat,
    pub lx: Dmat,
}

impl SpectroFinish {
    pub fn new(
        spectro: Spectro,
        freq: Dvec,
        f3qcm: F3qcm,
        f4qcm: F4qcm,
        irreps: Vec<Irrep>,
        lxm: Dmat,
        lx: Dmat,
    ) -> Self {
        Self {
            spectro,
            freq,
            f3qcm,
            f4qcm,
            irreps,
            lxm,
            lx,
        }
    }

    /// serialize `self` to JSON and write it to `filename`
    pub fn dump(&self, filename: &str) -> Result<(), Box<dyn Error>> {
        let f = std::fs::File::create(filename)?;
        serde_json::to_writer(f, self)?;
        Ok(())
    }

    /// load a [SpectroFinish] from the JSON file in `filename`
    pub fn load(filename: &str) -> Result<Self, Box<dyn Error>> {
        let r = std::fs::File::open(filename)?;
        Ok(serde_json::from_reader(r)?)
    }
}

impl Spectro {
    pub fn run(&self, deriv: Derivative) -> (Output, Restst) {
        let fc2 = deriv.fc2();
        // rotate the harmonic force constants to the new axes, and convert them
        // to the proper units
        let fc2 = self.rot2nd(fc2);
        let fc2 = FACT2 * fc2;

        // form the secular equations and decompose them to get harmonic
        // frequencies and the LXM matrix
        let w = self.geom.weights();
        let sqm: Vec<_> = w.iter().map(|w| 1.0 / w.sqrt()).collect();
        let fxm = self.form_sec(fc2, &sqm);
        let (harms, mut lxm) = symm_eigen_decomp(fxm, true);
        let freq = to_wavenumbers(&harms);

        // form the LX matrix
        let mut lx = self.make_lx(&sqm, &lxm);

        if self.rotor.is_sym_top() {
            self.bdegnl(&freq, &mut lxm, &w, &mut lx);
        }

        let irreps = compute_irreps(&self.geom, &lxm, self.nvib, 1e-4);

        if deriv.is_harmonic() {
            return (
                Output {
                    harms: freq.as_slice()[..self.nvib].to_vec(),
                    irreps,
                    rot_equil: self.rotcon.clone(),
                    geom: self.geom.clone(),
                    lxm: to_vec(lxm),
                    lx: to_vec(lx),
                    linear: self.rotor.is_linear(),
                    ..Default::default()
                },
                Restst::default(),
            );
        }

        let Derivative::Quartic(_, f3x, f4x) = deriv else {
	    panic!("can't handle cubics yet");
	};

        // start of cubic analysis
        let mut f3x = self.rot3rd(f3x);
        let f3qcm = force3(self.n3n, &mut f3x, &lx, self.nvib, &freq);

        // start of quartic analysis
        let f4x = self.rot4th(f4x);
        let f4qcm = force4(self.n3n, &f4x, &lx, self.nvib, &freq);

        self.finish(freq, f3qcm, f4qcm, irreps, lxm, lx)
    }

    /// finish the spectro run from F3qcm and F4qcm
    pub fn finish(
        &self,
        freq: Dvec,
        f3qcm: crate::f3qcm::F3qcm,
        f4qcm: crate::f4qcm::F4qcm,
        irreps: Vec<symm::Irrep>,
        lxm: Dmat,
        lx: Dmat,
    ) -> (Output, Restst) {
        let w = self.geom.weights();

        let (zmat, wila) = self.zeta(&lxm, &w);
        let quartic = Quartic::new(self, &freq, &wila);
        let restst = Restst::new(self, &zmat, &f3qcm, &freq);
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
                &f4qcm, &freq, &f3qcm, &zmat, fermi1, fermi2, modes, &wila,
            );
            (x, Some(g), e)
        } else {
            let (x, e) =
                self.xcalc(&f4qcm, &freq, &f3qcm, &zmat, modes, fermi1, fermi2);
            (x, None, e)
        };
        let (harms, funds) = if self.rotor.is_sym_top() {
            make_sym_funds(modes, &freq, &xcnst, &gcnst)
        } else {
            (
                freq.as_slice()[..self.nvib].to_vec(),
                make_funds(&freq, self.nvib, &xcnst),
            )
        };
        let rotnst = if self.rotor.is_sym_top() {
            self.alphas(&freq, &wila, &zmat, &f3qcm, modes, states, coriolis)
        } else {
            self.alphaa(&freq, &wila, &zmat, &f3qcm, modes, states, coriolis)
        };
        // this is worked on by resona and then enrgy so keep it out here
        let nstate = states.len();
        let mut eng = vec![0.0; nstate];
        if !self.rotor.is_sym_top() {
            resona(e0, modes, &freq, &xcnst, fermi1, fermi2, &mut eng);
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
        let (n1dm, n2dm, n3dm) = Mode::count(modes);
        for i in 1..n1dm + n2dm + n3dm + 1 {
            corrs.push(eng[i] - eng[0]);
        }
        // print_vib_states(&eng, &states);
        let rots = if self.rotor.is_sym_top() {
            if self.rotor.is_spherical_top() {
                panic!("don't know what to do with a spherical top here");
            }
            self.rots(&rotnst, states, &quartic)
        } else {
            self.rota(&rotnst, states, &quartic)
        };
        let sextic = Sextic::new(self, &wila, &zmat, &freq, &f3qcm);
        (
            Output {
                harms,
                funds,
                corrs,
                rots,
                irreps,
                quartic,
                sextic,
                rot_equil: self.rotcon.clone(),
                zpt: eng[0],
                geom: self.geom.clone(),
                lxm: to_vec(lxm),
                lx: to_vec(lx),
                linear: self.rotor.is_linear(),
            },
            restst,
        )
    }

    pub fn run_files<P>(
        &self,
        fort15: P,
        fort30: P,
        fort40: P,
    ) -> (Output, Restst)
    where
        P: AsRef<Path> + std::fmt::Debug,
    {
        let fc2 = load_fc2(fort15, self.n3n);
        let f3x = load_fc3(fort30, self.n3n);
        let f4x = load_fc4(fort40, self.n3n);

        self.run(Derivative::Quartic(fc2, f3x, f4x))
    }
}

fn to_vec(lxm: Dmat) -> Vec<Vec<f64>> {
    let (r, c) = lxm.shape();
    assert_eq!(r, c);
    let vlxm: Vec<Vec<f64>> =
        lxm.as_slice().chunks(r).map(|r| r.to_owned()).collect();
    vlxm
}

/// helper function for computing the irreps corresponding to `lxm`
pub fn compute_irreps(
    geom: &symm::Molecule,
    lxm: &crate::Dmat,
    nvib: usize,
    eps: f64,
) -> Vec<symm::Irrep> {
    let pg = geom.point_group_approx(eps);
    compute_irreps_in(geom, lxm, nvib, eps, pg)
}

/// compute irreps of vibrational modes in `pg`
pub fn compute_irreps_in(
    geom: &symm::Molecule,
    lxm: &Dmat,
    nvib: usize,
    eps: f64,
    pg: symm::PointGroup,
) -> Vec<Irrep> {
    let mut irreps = Vec::new();
    for (i, disp) in lxm.column_iter().take(nvib).enumerate() {
        let mol = geom.clone() + disp.as_slice().to_vec();
        let mut irrep = mol.irrep_approx(&pg, eps);
        let mut eps = eps;
        while let Err(e) = irrep {
            if eps >= 0.1 {
                eprintln!(
                    "failed to compute irrep {i} for\n{mol}\nin {pg} with {e:?}"
                );
                // give up and give A
                irrep = Ok(symm::Irrep::A);
                break;
            }
            eps *= 10.0;
            eprintln!("warning: raising epsilon to {eps:.1e}");
            irrep = mol.irrep_approx(&pg, eps);
        }
        irreps.push(irrep.unwrap());
    }
    irreps
}

use super::Output;

use std::path::Path;

use crate::consts::FACT2;
use crate::Mode;

use crate::resonance::Restst;
use crate::utils::force3;
use crate::utils::force4;
use crate::utils::load_fc2;
use crate::utils::load_fc3;
use crate::utils::load_fc4;
use crate::utils::make_funds;
use crate::utils::symm_eigen_decomp;
use crate::utils::to_wavenumbers;

use super::make_sym_funds;

use super::resona;

use crate::quartic::Quartic;

use crate::sextic::Sextic;

use super::Spectro;

impl Spectro {
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
        let (harms, mut lxm) = symm_eigen_decomp(fxm);
        let freq = to_wavenumbers(&harms);

        // form the LX matrix
        let mut lx = self.make_lx(&sqm, &lxm);

        if self.rotor.is_sym_top() {
            self.bdegnl(&freq, &mut lxm, &w, &mut lx);
        }

        // start of cubic analysis
        let f3x = load_fc3(fort30, self.n3n);
        let mut f3x = self.rot3rd(f3x);
        let f3qcm = force3(self.n3n, &mut f3x, &lx, self.nvib, &freq);

        // start of quartic analysis
        let f4x = load_fc4(fort40, self.n3n);
        let f4x = self.rot4th(f4x);
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
        let rots = if self.rotor.is_sym_top() {
            if self.rotor.is_spherical_top() {
                panic!("don't know what to do with a spherical top here");
            }
            self.rots(&rotnst, &states, &quartic)
        } else {
            self.rota(&rotnst, &states, &quartic)
        };

        Output {
            harms,
            funds,
            rots,
            corrs,
        }
    }
}

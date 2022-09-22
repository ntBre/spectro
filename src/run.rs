use super::{make_sym_funds, resona, Output, Spectro};
use crate::utils::{
    force3, force4, linalg::symm_eigen_decomp, load_fc2, load_fc3, load_fc4,
    make_funds, to_wavenumbers,
};
use crate::Mode;
use crate::{consts::FACT2, quartic::Quartic, resonance::Restst};
use std::path::Path;

impl Spectro {
    pub fn run<P>(self, fort15: P, fort30: P, fort40: P) -> Output
    where
        P: AsRef<Path> + std::fmt::Debug,
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
        let (harms, mut lxm) = symm_eigen_decomp(fxm, true);
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
                &f4qcm, &freq, &f3qcm, &zmat, fermi1, fermi2, modes, &wila,
            );
            (x, Some(g), e)
        } else {
            let (x, e) =
                self.xcalc(&f4qcm, &freq, &f3qcm, &zmat, modes, fermi1, fermi2);
            (x, None, e)
        };

        println!("xcnst={:.8}", xcnst);

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

        let quartic = Quartic::new(&self, &freq, &wila);
        let rots = if self.rotor.is_sym_top() {
            if self.rotor.is_spherical_top() {
                panic!("don't know what to do with a spherical top here");
            }
            self.rots(&rotnst, states, &quartic)
        } else {
            self.rota(&rotnst, states, &quartic)
        };

        Output {
            harms,
            funds,
            rots,
            corrs,
        }
    }
}

use std::cmp::{max, min};

use tensor::{Tensor3, Tensor4};

use crate::{
    utils::{find3r, ioff, make_tau},
    Dmat, Dvec, Spectro, Vec3, SQLAM,
};

/// struct holding the sextic distortion constants
pub(crate) struct Sextic {}

impl Sextic {
    pub(crate) fn new(
        spectro: &Spectro,
        nvib: usize,
        wila: &Dmat,
        zmat: &Tensor3,
        freq: &Dvec,
        f3qcm: &[f64],
        rotcon: &[f64],
    ) -> Self {
        // convert to Hz from cm⁻¹
        const CONST2: f64 = 2.99792458e10;
        let maxcor = if spectro.is_linear.unwrap() { 2 } else { 3 };
        let primat = spectro.geom.principal_moments();
        let c = c_mat(maxcor, nvib, freq, &primat, wila);
        let t = t_mat(maxcor, nvib, freq, &c);
        let cc = cc_tensor(nvib, maxcor, freq, &c, zmat, rotcon);
        let tau = make_tau(maxcor, nvib, freq, &primat, wila);
        let scc = scc(maxcor, tau, rotcon, nvib, freq, cc, f3qcm, &c, spectro);
        Sextic {}
    }
}

fn scc(
    maxcor: usize,
    tau: Tensor4,
    rotcon: &[f64],
    nvib: usize,
    freq: &Dvec,
    cc: Tensor4,
    f3qcm: &[f64],
    c: &Dmat,
    spectro: &Spectro,
) -> Tensor3 {
    let mut scc = Tensor3::zeros(3, 3, 3);
    for ixyz in 1..=maxcor {
        let iixyz = ioff(ixyz + 1);

        let mut val1 = 0.0;
        for jxyz in 1..=maxcor {
            val1 += tau[(jxyz - 1, ixyz - 1, ixyz - 1, ixyz - 1)].powi(2)
                / rotcon[jxyz - 1];
        }
        val1 = 3.0 * val1 / 16.0;

        let mut val2 = 0.0;
        for i in 0..nvib {
            val2 += freq[i] * cc[(i, ixyz - 1, ixyz - 1, ixyz - 1)].powi(2);
        }
        val2 *= 2.0;

        let mut val3 = 0.0;
        for i in 0..nvib {
            for j in 0..nvib {
                for k in 0..nvib {
                    let ijk = find3r(i, j, k);
                    val3 += f3qcm[ijk]
                        * c[(i, iixyz - 1)]
                        * c[(j, iixyz - 1)]
                        * c[(k, iixyz - 1)];
                }
            }
        }
        val3 /= 6.0;

        if spectro.is_linear.unwrap() {
            todo!() // goto 510
        };

        const TOL: f64 = 1e-4;
        let mut val4 = 0.0;
        for jxyz in 1..=3 {
            if ixyz != jxyz {
                let div = rotcon[ixyz - 1] - rotcon[jxyz - 1];
                if div.abs() > TOL {
                    val4 += tau[(jxyz - 1, ixyz - 1, ixyz - 1, ixyz - 1)]
                        .powi(2)
                        / div;
                }
            }
        }
        val4 *= 0.25;

        let value = val1 - val2 + val3 + val4;
        scc[(ixyz - 1, ixyz - 1, ixyz - 1)] = value;
    }
    scc
}

fn cc_tensor(
    nvib: usize,
    maxcor: usize,
    freq: &Dvec,
    c: &Dmat,
    zmat: &Tensor3,
    rotcon: &[f64],
) -> Tensor4 {
    let mut cc = Tensor4::zeros(nvib, 3, 3, 3);
    for i in 0..nvib {
        for ixyz in 1..=maxcor {
            let iixyz = ioff(ixyz + 1);

            let mut val = 0.0;
            for j in 0..nvib {
                let val1 = 1.0 / freq[j].powi(2) + 2.0 / freq[i].powi(2);
                let val2 = c[(j, iixyz - 1)]
                    * zmat[(j, i, ixyz - 1)]
                    * rotcon[ixyz - 1]
                    * freq[j].powf(1.5);
                val += val2 * val1 / freq[i].sqrt();
            }
            cc[(i, ixyz - 1, ixyz - 1, ixyz - 1)] = val;

            for jxyz in 1..=maxcor {
                if ixyz != jxyz {
                    let ijxyz = ioff(max(ixyz, jxyz)) + min(ixyz, jxyz);

                    let mut val = 0.0;
                    for j in 0..nvib {
                        let val1 =
                            1.0 / freq[j].powi(2) + 2.0 / freq[i].powi(2);
                        let val2 = freq[j].powf(1.5)
                            * (c[(j, iixyz - 1)]
                                * zmat[(j, i, jxyz - 1)]
                                * rotcon[jxyz - 1]
                                + 2.0
                                    * c[(j, ijxyz - 1)]
                                    * zmat[(j, i, ixyz - 1)]);
                        val += val2 * val1 / freq[i].sqrt();
                    }
                    cc[(i, jxyz - 1, ixyz - 1, ixyz - 1)] = val;
                    cc[(i, ixyz - 1, jxyz - 1, ixyz - 1)] = val;
                    cc[(i, ixyz - 1, ixyz - 1, jxyz - 1)] = val;

                    for kxyz in 1..=maxcor {
                        if kxyz != jxyz {
                            if kxyz != ixyz {
                                let ikxyz =
                                    ioff(max(ixyz, kxyz)) + min(ixyz, kxyz);
                                let jkxyz =
                                    ioff(max(jxyz, kxyz)) + min(jxyz, kxyz);

                                let mut val = 0.0;
                                for j in 0..nvib {
                                    let val1 = 1.0 / freq[j].powi(2)
                                        + 2.0 / freq[i].powi(2);
                                    let val2 = freq[j].powf(1.5)
                                        * (c[(j, jkxyz - 1)]
                                            * zmat[(j, i, ixyz - 1)]
                                            * rotcon[ixyz - 1]
                                            + c[(j, ikxyz - 1)]
                                                * zmat[(j, i, jxyz - 1)]
                                                * rotcon[jxyz - 1]
                                            + c[(j, ijxyz - 1)]
                                                * zmat[(j, i, kxyz - 1)]
                                                * rotcon[kxyz - 1]);
                                    val += val2 * val1 / freq[i].sqrt();
                                }
                                cc[(i, kxyz - 1, jxyz - 1, ixyz - 1)] = val;
                            }
                        }
                    }
                }
            }
        }
    }
    cc
}

// went with the fortran indexing here and then subtracting one since that was
// easier than figuring out the normal indexing
fn t_mat(maxcor: usize, nvib: usize, freq: &Dvec, c: &Dmat) -> Dmat {
    let mut t = Dmat::zeros(maxcor, maxcor);
    for ixyz in 1..=maxcor {
        let iixyz = ioff(ixyz + 1);
        for jxyz in 1..=maxcor {
            let ijxyz = ioff(max(ixyz, jxyz)) + min(ixyz, jxyz);
            let jjxyz = ioff(jxyz + 1);
            let mut val = 0.0;
            for i in 0..nvib {
                val += freq[i] * c[(i, iixyz - 1)] * c[(i, jjxyz - 1)];
                if ixyz != jxyz {
                    val +=
                        2.0 * freq[i] * c[(i, ijxyz - 1)] * c[(i, ijxyz - 1)];
                }
            }
            t[(ixyz - 1, jxyz - 1)] = -0.5 * val;
        }
    }
    t
}

fn c_mat(
    maxcor: usize,
    nvib: usize,
    freq: &Dvec,
    primat: &Vec3,
    wila: &Dmat,
) -> Dmat {
    let mut c = Dmat::zeros(maxcor, 6);
    for i in 0..nvib {
        let cnst = (SQLAM * SQLAM * freq[i]).powf(1.5);
        for ixyz in 0..maxcor {
            for jxyz in 0..=ixyz {
                let ijxyz = ioff(ixyz + 1) + jxyz;
                let div = 2.0 * primat[ixyz] * primat[jxyz] * cnst;
                c[(i, ijxyz)] = wila[(i, ijxyz)] / div;
            }
        }
    }
    c
}

#[cfg(test)]
mod tests {
    use approx::assert_abs_diff_eq;
    use nalgebra::dmatrix;

    use crate::{
        rotor::Rotor,
        utils::{
            force3, force4, funds, load_fc2, load_fc3, load_fc4,
            symm_eigen_decomp, to_wavenumbers, xcalc,
        },
        CONST, FACT2,
    };

    use super::*;

    #[test]
    fn test_all() {
        let mut spectro = Spectro::load("testfiles/h2o.in");
        spectro.geom.to_angstrom();
        spectro.geom.normalize();
        let axes = spectro.geom.reorder();
        let rotor = spectro.rotor_type();
        let natom = spectro.natoms();
        let n3n = 3 * natom;
        let nvib = n3n - 6
            + if let Rotor::Linear = rotor {
                spectro.is_linear = Some(true);
                1
            } else {
                spectro.is_linear = Some(false);
                0
            };
        let i3vib = nvib * (nvib + 1) * (nvib + 2) / 6;
        let i4vib = nvib * (nvib + 1) * (nvib + 2) * (nvib + 3) / 24;
        let fc2 = load_fc2("testfiles/fort.15", n3n);
        let fc2 = spectro.rot2nd(fc2, axes);
        let fc2 = FACT2 * fc2;
        let w = spectro.geom.weights();
        let sqm: Vec<_> = w.iter().map(|w| 1.0 / w.sqrt()).collect();
        let fxm = spectro.form_sec(fc2, n3n, &sqm);
        let (harms, lxm) = symm_eigen_decomp(fxm);
        let freq = to_wavenumbers(harms);
        let lx = spectro.make_lx(n3n, &sqm, &lxm);
        let (zmat, _biga, wila) = spectro.zeta(natom, nvib, &lxm, &w);
        let f3x = load_fc3("testfiles/fort.30", n3n);
        let mut f3x = spectro.rot3rd(n3n, natom, f3x, axes);
        let f3qcm = force3(n3n, &mut f3x, &lx, nvib, &freq, i3vib);
        let f4x = load_fc4("testfiles/fort.40", n3n);
        let mut f4x = spectro.rot4th(n3n, natom, f4x, axes);
        let f4qcm = force4(n3n, &mut f4x, &lx, nvib, &freq, i4vib);
        let moments = spectro.geom.principal_moments();
        let rotcon: Vec<_> = moments.iter().map(|m| CONST / m).collect();
        let (xcnst, _e0) = xcalc(nvib, &f4qcm, &freq, &f3qcm, &zmat, &rotcon);
        let fund = funds(&freq, nvib, &xcnst);

        let primat = spectro.geom.principal_moments();

        // c_mat
        let c = {
            let got = c_mat(3, 3, &freq, &primat, &wila);
            // sign of second element is swapped but that's okay since wila
            // comes from LXM
            let want = dmatrix![
                 0.00000000, -0.00070197, 0.00000000, 0.00000000, -0.00000000,
            0.00000000;
            -0.00098872, 0.00000000, -0.00053911, -0.00000000, 0.00000000,
            -0.00034891;
            -0.00488437, 0.00000000, 0.00136650,
                -0.00000000, 0.00000000, -0.00001189;
                           ];
            assert_abs_diff_eq!(got, want, epsilon = 1e-8);
            got
        };
        // t_mat
        {
            let got = t_mat(3, 3, &freq, &c);
            let want = dmatrix! {
            -0.02156707,  0.00254453, -0.00070920;
             0.00254453, -0.00209853, -0.00034716;
            -0.00070920, -0.00034716, -0.00023348;
            };
            assert_abs_diff_eq!(got, want, epsilon = 5e-8);
        }
    }
}

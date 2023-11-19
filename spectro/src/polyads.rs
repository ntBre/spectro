#![allow(unused)]

use super::{Dmat, Dvec, Tensor3};
use crate::f3qcm::F3qcm;
use crate::f4qcm::F4qcm;
use crate::resonance::Fermi1;
use crate::resonance::Fermi2;
use crate::utils::find4;
use crate::Mode;
use nalgebra::DMatrix;
use std::collections::HashSet;

/// set up resonance polyad matrices for asymmetric tops and compute their
/// eigenvalues and eigenvectors
pub(crate) fn resona(
    zmat: &Tensor3,
    f3qcm: &F3qcm,
    f4qcm: &F4qcm,
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
    let dnm = init_res_denom(n1dm, freq, fermi1, fermi2);

    let mut zpe = e0;
    for ii in 0..n1dm {
        let i = i1mode[ii];
        zpe += freq[i] * 0.5;
        for jj in 0..=ii {
            let j = i1mode[jj];
            zpe += xcnst[(i, j)] * 0.25;
        }
    }

    // TODO handle separate resonance blocks. the example in the comments is one
    // for each symmetry in C2v, ie a1, a2, b1, and b2 symmetries
    // let iirst = make_resin(fermi1, n1dm, fermi2);

    // TODO generate this! this is only for debugging to match the order from
    // spectro2.in for c2h4 in old-spectro
    let iirst = nalgebra::dmatrix![
        0,    0,    1,    0,    0,    0,    0,    0,    0,    0,    0,    0;
        0,    0,    0,    0,    1,    0,    0,    0,    0,    0,    0,    0;
        0,    0,    0,    1,    0,    0,    0,    0,    0,    0,    0,    0;
        0,    0,    0,    0,    1,    1,    0,    0,    0,    0,    0,    0;
        0,    0,    0,    0,    1,    0,    1,    0,    0,    0,    0,    0;
        0,    0,    0,    0,    2,    0,    0,    0,    0,    0,    0,    0;
        0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    2;
        0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0;
        0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0;
        0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0;
        0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0;
        0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0;
    ];
    // transpose to match fortran indexing
    let iirst = iirst.transpose();

    let (nreson, _) = iirst.shape();
    for ist in 0..nreson {
        let mut e = e0;
        for ii in 0..n1dm {
            let i = i1mode[ii];
            e += freq[i] * (iirst[(ii, ist)] as f64 + 0.5);
            for jj in 0..=ii {
                let j = i1mode[jj];
                e += xcnst[(i, j)]
                    * (iirst[(ii, ist)] as f64 + 0.5)
                    * (iirst[(jj, ist)] as f64 + 0.5);
            }
        }
        eng[ist] = e - zpe;
        eprintln!("E* of state {ist} = {}", eng[ist]);
        // NOTE I have all the right numbers, but the order is different and
        // differs each time. I may need to revisit the usage of a HashMap for
        // the resonances at some point.
    }

    // let idimen = nreson * (nreson + 1) / 2;
    let mut resmat = DMatrix::zeros(nreson, nreson);
    for i in 0..nreson {
        for j in 0..=i {
            if j == i {
                resmat[(i, j)] = eng[i];
            } else {
                resmat[(i, j)] = genrsa(
                    n1dm, zmat, f3qcm, f4qcm, &iirst, i, j, &i1mode, freq, &dnm,
                );
            }
        }
    }
    // construct the resonance matrix, then call symm_eigen_decomp to get the
    // eigenvalues and eigenvectors
    println!("resmat={:.8}", resmat);
}

/// computes the general resonance element between states `istate` and `jstate`
/// for an asymmetric top. `iirst` is a matrix containing the quantum numbers of
/// the states involved in the resonance polyad
pub(crate) fn genrsa(
    n1dm: usize,
    zmat: &Tensor3,
    f3qcm: &F3qcm,
    f4qcm: &F4qcm,
    iirst: &DMatrix<usize>,
    istate: usize,
    jstate: usize,
    i1mode: &[usize],
    freq: &Dvec,
    dnm: &Tensor3,
) -> f64 {
    let mut idiff = 0;
    let mut ndelta = 0;
    let mut ndel = 0;
    let mut nleft = Vec::new();
    let mut nright = Vec::new();
    let mut ndiff = Vec::new();
    let mut nmin = Vec::new();
    let mut indx = Vec::new();
    for i in 0..n1dm {
        let nnleft = iirst[(i, istate)];
        let nnright = iirst[(i, jstate)];
        let ndiffer = nnright as isize - nnleft as isize;
        if ndiffer != 0 {
            ndelta += ndiffer.abs();
            ndel += ndiffer;
            idiff += 1;
            if idiff > 4 || ndelta > 4 {
                eprintln!("higher than quartic resonances not yet implemented");
                eprintln!("setting resonance constant to zero");
                return 0.0;
            }
            nleft.push(nnleft);
            nright.push(nnright);
            ndiff.push(ndiffer);
            nmin.push(nnleft.min(nnright));
            indx.push(i);
        }
    }

    match idiff {
        4 => {
            if ndel == 0 {
                // this is in a fortran equivalence block, which should tie each of
                // these variables permanently I think, so I'll probably have to
                // write this everywhere I need these indices
                let ii = indx[0];
                let jj = indx[1];
                let kk = indx[2];
                let ll = indx[3];

                let na = nmin[0];
                let nb = nmin[1];
                let nc = nmin[2];
                let nd = nmin[3];
                // case 1a: Kabcd
                res2a(zmat, f3qcm, f4qcm, i1mode, freq, dnm, ii, jj, kk, ll)
                    * (((na + 1) * (nb + 1) * (nc + 1) * (nd + 1)) as f64 / 16.)
                        .sqrt()
            } else {
                // case 1b: Ka,bcd resonance
                // sort indices in required order first
                todo!("case 1b: Ka,bcd");
            }
        }
        3 => todo!(),
        2 => match ndelta {
            4 => todo!(),
            3 => todo!(),
            2 => {
                // case 4: Lehmann's "1-1" resonance
                let ii = indx[0];
                let jj = indx[1];
                let na = nmin[0];
                let nb = nmin[1];
                let val1 =
                    res2a(
                        zmat, f3qcm, f4qcm, i1mode, freq, dnm, ii, ii, ii, jj,
                    ) * f64::sqrt(((na + 1).pow(3) * (nb + 1)) as f64 / 16.0);
                let val2 =
                    res2a(
                        zmat, f3qcm, f4qcm, i1mode, freq, dnm, jj, jj, jj, ii,
                    ) * f64::sqrt(((nb + 1).pow(3) * (na + 1)) as f64 / 16.0);
                let mut val3 = 0.0;
                for k in 0..n1dm {
                    if k != ii && k != jj {
                        let nk1 = iirst[(k, istate)];
                        let nk2 = iirst[(k, jstate)];
                        if nk1 != nk2 {
                            panic!("!!!Internal error in genrsa!!! nk1={nk1} nk2={nk2}");
                        }
                        val3 +=
                            2. * res2a(
                                zmat, f3qcm, f4qcm, &i1mode, freq, dnm, ii, k,
                                jj, k,
                            ) * f64::sqrt(
                                dble((na + 1) * (nb + 1))
                                    * (dble(nk1) + 0.5).powi(2)
                                    / 16.,
                            );
                    }
                }
                val1 + val2 + val3
            }
            _ => 0.0,
        },
        _ => 0.0,
    }
}

#[inline]
const fn dble(n: usize) -> f64 {
    n as f64
}

pub(crate) fn res2a(
    zmat: &tensor::Tensor3<f64>,
    f3qcm: &F3qcm,
    f4qcm: &F4qcm,
    i1mode: &[usize],
    freq: &Dvec,
    dnm: &Tensor3,
    ii: usize,
    jj: usize,
    kk: usize,
    ll: usize,
) -> f64 {
    let d = Denom { freq, dnm };
    // I sure hope this isn't the same indx from outside
    let indx = [ii, jj, kk, ll];
    let n1dm = i1mode.len();
    use Sign::*;
    let mut case = "????";
    if ii == jj && kk == ll {
        todo!()
    } else if ii == jj && ii == kk {
        case = "aaab";
        let i = i1mode[ii];
        let j = i1mode[ll];
        let val1 = f4qcm[(i, i, i, j)] / 2.0;
        let mut val2 = 0.0;
        let mut val3 = 0.0;
        let mut val4 = 0.0;
        for mm in 0..n1dm {
            let k = i1mode[mm];
            let iik = f3qcm[(i, i, k)];
            let ijk = f3qcm[(i, j, k)];
            let temp = -0.5
                * (-4.0 / freq[k as usize]
                    + d.denom(Plus(i), Plus(i), Minus(k))
                    + d.denom(Minus(i), Minus(i), Minus(k)));
            val3 -= 0.25 * iik * ijk * temp;
            let temp = -0.5
                * (2.0
                    * (d.denom(Plus(i), Minus(j), Minus(k))
                        + d.denom(Minus(i), Plus(j), Minus(k)))
                    + d.denom(Plus(i), Plus(j), Minus(k))
                    + d.denom(Minus(i), Minus(j), Minus(k)));

            val4 -= 0.25 * iik * ijk * temp;
        }
        // the sign I'm getting is backwards but I think that's okay
        dbg!(val1 + val2 + val3 + val4)
    } else {
        todo!()
    }
}

enum Sign {
    Plus(usize),
    Minus(usize),
}

impl Sign {
    fn abs(&self) -> usize {
        match self {
            Self::Plus(n) | Self::Minus(n) => *n,
        }
    }

    fn signum(&self) -> isize {
        match self {
            Sign::Plus(_) => 1,
            Sign::Minus(_) => -1,
        }
    }
}

struct Denom<'a> {
    freq: &'a Dvec,
    dnm: &'a Tensor3,
}

impl Denom<'_> {
    fn denom(&self, is: Sign, js: Sign, ks: Sign) -> f64 {
        let i = is.abs();
        let j = js.abs();
        let k = ks.abs();
        // these are computed in fortran as is/i so signum should be fine. if any of
        // them were zero, the division would obviously be disastrous so they must
        // be non-zero
        let isign = is.signum();
        let jsign = js.signum();
        let ksign = ks.signum();
        let iglobsgn = isign * jsign * ksign;
        let inneg = (3 - isign - jsign - ksign) / 2;
        let isign = isign * iglobsgn;
        let jsign = jsign * iglobsgn;
        let ksign = ksign * iglobsgn;
        if inneg == 0 || inneg == 3 {
            1. / (self.freq[i] + self.freq[j] + self.freq[k]) * iglobsgn as f64
        } else if isign > 0 {
            self.dnm[(i, j, k)] * iglobsgn as f64
        } else if jsign > 0 {
            self.dnm[(j, i, k)] * iglobsgn as f64
        } else {
            self.dnm[(k, i, j)] * iglobsgn as f64
        }
    }
}

/// initialize the inverse resonance denominators and zero any elements
/// corresponding to Fermi resonances deleted in the contact transformation
pub(crate) fn init_res_denom(
    n1dm: usize,
    freq: &Dvec,
    fermi1: &[Fermi1],
    fermi2: &[Fermi2],
) -> Tensor3 {
    let n = n1dm;
    let mut dnom = Tensor3::zeros(n, n, n);
    for i in 0..n {
        for j in 0..n {
            for k in 0..n {
                dnom[(i, j, k)] = 1.0 / (freq[i] - freq[j] - freq[k]);
            }
        }
    }

    for &Fermi1 { i, j } in fermi1 {
        dnom[(j, i, i)] = 0.0;
    }

    for &Fermi2 { i: j, j: k, k: i } in fermi2 {
        dnom[(i, j, k)] = 0.0;
        dnom[(i, k, j)] = 0.0;
    }
    dnom
}

/// construct the RESIN Fermi polyad matrix. NOTE that the comments in resona.f
/// mention multiple blocks for different symmetries. However, the only way we
/// use it is with a single block, so I'm writing this code with that in mind.
pub(crate) fn make_resin(
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
    let data: Vec<_> = data.into_iter().flatten().collect();
    DMatrix::<usize>::from_row_slice(data.len() / n1dm, n1dm, &data)
}

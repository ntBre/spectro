#![allow(unused)]

use super::{Dmat, Dvec, Tensor3};
use crate::f3qcm::F3qcm;
use crate::f4qcm::F4qcm;
use crate::resonance::Fermi1;
use crate::resonance::Fermi2;
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
    let _dnom = init_res_denom(n1dm, freq, fermi1, fermi2);

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
        eprintln!("E* of state {ist} = {}", eng[ist]);
        // NOTE I have all the right numbers, but the order is different and
        // differs each time. I may need to revisit the usage of a HashMap for
        // the resonances at some point.
    }

    // println!("iirst={:.8}", iirst);
    // let idimen = nreson * (nreson + 1) / 2;
    let mut resmat = DMatrix::zeros(nreson, nreson);
    for i in 0..nreson {
        for j in 0..=i {
            if j == i {
                resmat[(i, j)] = eng[i];
            } else {
                resmat[(i, j)] = genrsa(zmat, f3qcm, f4qcm, &iirst, i, j);
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
    _zmat: &Tensor3,
    _f3qcm: &F3qcm,
    _f4qcm: &F4qcm,
    iirst: &DMatrix<usize>,
    istate: usize,
    jstate: usize,
) -> f64 {
    dbg!(istate, jstate);
    let mut idiff = 0;
    let mut ndelta = 0;
    let mut ndel = 0;
    let mut nleft = Vec::new();
    let mut nright = Vec::new();
    let mut ndiff = Vec::new();
    let mut nmin = Vec::new();
    let mut indx = Vec::new();
    let (n1dm, _) = iirst.shape();
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

    if idiff == 4 {
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
            return res2a(_zmat, _f3qcm, _f4qcm, ii, jj, kk, ll)
                * (((na + 1) * (nb + 1) * (nc + 1) * (nd + 1)) as f64 / 16.)
                    .sqrt();
            // TODO this might not return yet, but it does set the function name
        } else {
            // case 1b: Ka,bcd resonance
            // sort indices in required order first
            eprintln!("todo case 1b: Ka,bcd");
            return 0.0;
        }
    }
    0.0
}

pub(crate) fn res2a(
    _zmat: &tensor::Tensor3<f64>,
    _f3qcm: &F3qcm,
    _f4qcm: &F4qcm,
    ii: usize,
    jj: usize,
    kk: usize,
    ll: usize,
) -> f64 {
    todo!()
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

    for &Fermi2 { i, j, k } in fermi2 {
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

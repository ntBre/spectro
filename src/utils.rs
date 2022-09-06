use std::{
    f64::consts::SQRT_2,
    fmt::{Debug, Display},
    fs::read_to_string,
    iter::zip,
    path::Path,
    str::FromStr,
};

use nalgebra::{dmatrix, SymmetricEigen};
use tensor::Tensor4;
type Tensor3 = tensor::tensor3::Tensor3<f64>;

use crate::{
    f3qcm::F3qcm, f4qcm::F4qcm, ifrm1::Ifrm1, Dmat, Dvec, Mode, Spectro, FACT3,
    FACT4, FUNIT3, FUNIT4, ICTOP, IPTOC, WAVE,
};

// separate for macro
use crate::f4qcm;

impl Display for Spectro {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        use crate::Curvil::*;
        writeln!(f, "# SPECTRO #############")?;
        for chunk in self.header.chunks(15) {
            for i in chunk {
                write!(f, "{:5}", i)?;
            }
            writeln!(f)?;
        }
        writeln!(f, "# GEOM #############")?;
        writeln!(f, "{:5}{:5}", self.geom.atoms.len() + self.dummies.len(), 1)?;
        for atom in &self.geom.atoms {
            writeln!(
                f,
                "{:5.2}{:16.8}{:16.8}{:16.8}",
                atom.atomic_number as f64, atom.x, atom.y, atom.z
            )?;
        }
        for dummy in &self.dummies {
            let atom = dummy.get_vals(&self.geom);
            writeln!(
                f,
                "{:5.2}{:16.8}{:16.8}{:16.8}",
                0.0, atom[0], atom[1], atom[2],
            )?;
        }
        writeln!(f, "# WEIGHT #############")?;
        writeln!(f, "{:5}", self.weights.len())?;
        for weight in &self.weights {
            writeln!(f, "{:5}{:12.6}", weight.0, weight.1)?;
        }
        writeln!(f, "# CURVIL #############")?;
        for curvil in &self.curvils {
            match curvil {
                Bond(i, j) => write!(f, "{:5}{:5}", i, j)?,
                Bend(i, j, k) => write!(f, "{:5}{:5}{:5}", i, j, k)?,
                Tors(i, j, k, l) => write!(f, "{:5}{:5}{:5}{:5}", i, j, k, l)?,
            }
            writeln!(f)?;
        }
        if !self.degmodes.is_empty() {
            writeln!(f, "# DEGMODE #############")?;
            for curvil in &self.degmodes {
                for i in curvil {
                    write!(f, "{:5}", i)?;
                }
                writeln!(f)?;
            }
        }
        Ok(())
    }
}

/// parse an entire `line` into a vector of the same type
pub(crate) fn parse_line<T: FromStr>(line: &str) -> Vec<T>
where
    <T as FromStr>::Err: Debug,
{
    line.split_whitespace()
        .map(|s| s.parse::<T>().expect(&format!("failed to parse {}", s)))
        .collect::<Vec<_>>()
}

/// helper for sorting the find3r and find4t indices
fn sort_indices<const N: usize>(mut indices: [usize; N]) -> [usize; N] {
    indices.sort();
    indices
}

/// cubic force constant indexing formula. I think it relies on the fortran
/// numbering though, so I need to add one initially and then subtract one at
/// the end. this returns *indices* so it doesn't work directly for computing
/// lengths. For example, `find3r(3, 3, 3) = 19` and `find3r(2, 2, 2) = 9`, so
/// to get the length of the cubic force constant vector for water you have to
/// do `find3r(2, 2, 2) + 1`, where 2 is `nvib-1`
pub(crate) fn find3(i: usize, j: usize, k: usize) -> usize {
    let [i, j, k] = sort_indices([i + 1, j + 1, k + 1]);
    i + (j - 1) * j / 2 + (k - 1) * k * (k + 1) / 6 - 1
}

/// quartic force constant indexing formula. it relies on the fortran numbering,
/// so I need to add one initially and then subtract one at the end
pub(crate) fn find4(i: usize, j: usize, k: usize, l: usize) -> usize {
    let [i, j, k, l] = sort_indices([i + 1, j + 1, k + 1, l + 1]);
    i + (j - 1) * j / 2
        + (k - 1) * k * (k + 1) / 6
        + (l - 1) * l * (l + 1) * (l + 2) / 24
        - 1
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
pub fn to_wavenumbers(freqs: &Dvec) -> Dvec {
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

pub fn load_fc2<P>(infile: P, n3n: usize) -> Dmat
where
    P: AsRef<Path>,
{
    let data = read_to_string(infile).unwrap();
    Dmat::from_iterator(
        n3n,
        n3n,
        data.split_whitespace().map(|s| s.parse().unwrap()),
    )
}

pub fn load_fc3<P>(infile: P, n3n: usize) -> Tensor3
where
    P: AsRef<Path>,
{
    let mut f3x = Tensor3::zeros(n3n, n3n, n3n);
    let f33 = load_vec(infile);
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

pub fn load_fc4<P>(infile: P, n3n: usize) -> Tensor4
where
    P: AsRef<Path>,
{
    let mut f4x = Tensor4::zeros(n3n, n3n, n3n, n3n);
    let f44 = load_vec(infile);
    let mut mabc = 0;
    for iabc in 0..n3n {
        for jabc in 0..=iabc {
            for kabc in 0..=jabc {
                for labc in 0..=kabc {
                    let val = f44[mabc];
                    f4x[(iabc, jabc, kabc, labc)] = val;
                    f4x[(iabc, jabc, labc, kabc)] = val;
                    f4x[(iabc, kabc, jabc, labc)] = val;
                    f4x[(iabc, kabc, labc, jabc)] = val;
                    f4x[(iabc, labc, jabc, kabc)] = val;
                    f4x[(iabc, labc, kabc, jabc)] = val;
                    f4x[(jabc, iabc, kabc, labc)] = val;
                    f4x[(jabc, iabc, labc, kabc)] = val;
                    f4x[(jabc, kabc, iabc, labc)] = val;
                    f4x[(jabc, kabc, labc, iabc)] = val;
                    f4x[(jabc, labc, iabc, kabc)] = val;
                    f4x[(jabc, labc, kabc, iabc)] = val;
                    f4x[(kabc, iabc, jabc, labc)] = val;
                    f4x[(kabc, iabc, labc, jabc)] = val;
                    f4x[(kabc, jabc, iabc, labc)] = val;
                    f4x[(kabc, jabc, labc, iabc)] = val;
                    f4x[(kabc, labc, iabc, jabc)] = val;
                    f4x[(kabc, labc, jabc, iabc)] = val;
                    f4x[(labc, iabc, jabc, kabc)] = val;
                    f4x[(labc, iabc, kabc, jabc)] = val;
                    f4x[(labc, jabc, iabc, kabc)] = val;
                    f4x[(labc, jabc, kabc, iabc)] = val;
                    f4x[(labc, kabc, iabc, jabc)] = val;
                    f4x[(labc, kabc, jabc, iabc)] = val;
                    mabc += 1;
                }
            }
        }
    }
    f4x
}

pub(crate) fn load_vec<P: AsRef<Path>>(infile: P) -> Vec<f64> {
    let data = read_to_string(infile).unwrap();
    data.split_ascii_whitespace()
        .map(|s| s.parse().unwrap())
        .collect()
}

pub(crate) fn close(a: f64, b: f64, eps: f64) -> bool {
    (a - b).abs() < eps
}

/// freq is the vector of harmonic frequencies
pub fn force3(
    n3n: usize,
    f3x: &mut Tensor3,
    lx: &Dmat,
    nvib: usize,
    freq: &Dvec,
) -> F3qcm {
    for kabc in 0..n3n {
        let start = (0, 0);
        let end = (n3n, n3n - 1);
        let mut dd =
            Dmat::from_row_slice(n3n, n3n, &f3x.submatrix(start, end, kabc));
        dd *= FUNIT3;
        let ee = lx.clone().transpose() * dd.clone() * lx.clone();
        f3x.set_submatrix(start, end, kabc, ee.data.as_slice());
    }
    let mut f3qcm =
        F3qcm::with_capacity(find3(nvib - 1, nvib - 1, nvib - 1) + 1);
    for i in 0..nvib {
        let wi = freq[(i)];
        for j in 0..=i {
            let wj = freq[(j)];
            for k in 0..=j {
                let wk = freq[(k)];
                let wijk = wi * wj * wk;
                let fact = FACT3 / wijk.sqrt();
                let mut val = 0.0;
                for l in 0..n3n {
                    val += f3x[(i, j, l)] * lx[(l, k)];
                }
                f3qcm.push(val * fact);
            }
        }
    }
    f3qcm
}

pub fn force4(
    n3n: usize,
    f4x: &Tensor4,
    lx: &Dmat,
    nvib: usize,
    harms: &Dvec,
) -> F4qcm {
    let lxt = lx.transpose();
    let mut f4q = Tensor4::zeros(n3n, n3n, n3n, n3n);
    for kabc in 0..n3n {
        for labc in 0..n3n {
            let mut dd = Dmat::from_row_slice(
                n3n,
                n3n,
                &f4x.submatrix((0, 0), (n3n, n3n - 1), kabc, labc),
            );
            dd *= FUNIT4;
            let ee = lxt.clone() * dd * lx.clone();
            // technically this as_slice call is wrong because it comes out in
            // column-major order, but the matrix is symmetric
            f4q.set_submatrix(
                (0, 0),
                (n3n, n3n - 1),
                kabc,
                labc,
                ee.data.as_slice(),
            );
        }
    }
    // now can I include the loop above in here - the holy grail
    let n = nvib - 1;
    let mut f4qcm = f4qcm![0.0; find4(n, n, n, n) + 1];
    for ii in 0..nvib {
        let wi = harms[ii];
        for jj in 0..=ii {
            let wj = harms[jj];
            let mut dd = Dmat::zeros(n3n, n3n);
            for kabc in 0..n3n {
                for labc in 0..n3n {
                    dd[(kabc, labc)] = f4q[(ii, jj, kabc, labc)];
                }
            }
            let ee = lxt.clone() * dd * lx.clone();
            for ivib in 0..nvib {
                let wk = harms[ivib];
                for jvib in 0..=ivib {
                    let wl = harms[jvib];
                    let wijkl = wi * wj * wk * wl;
                    let sqws = wijkl.sqrt();
                    let fact = FACT4 / sqws;
                    let ijkl = (ivib, jvib, ii, jj);
                    f4qcm[ijkl] = ee[(ivib, jvib)] * fact;
                }
            }
        }
    }
    f4qcm
}

/// make E0 for asymmetric tops or the first component of E0 for symmetric tops
pub(crate) fn make_e0(
    modes: &[Mode],
    f4qcm: &F4qcm,
    f3qcm: &F3qcm,
    freq: &Dvec,
    ifrm1: &Ifrm1,
    ifrmchk: &tensor::Tensor3<usize>,
) -> f64 {
    // NOTE: took out some weird IA stuff here and reproduced their results.
    // maybe my signs are actually right and theirs are wrong.
    let mut f4k = 0.0;
    let mut f3k = 0.0;
    let mut f3kkl = 0.0;
    let (n1dm, _, _) = Mode::count(modes);
    let (i1mode, _, _) = Mode::partition(modes);
    for kk in 0..n1dm {
        let k = i1mode[kk];
        // kkkk and kkk terms
        let fiqcm = f4qcm[(k, k, k, k)];
        f4k += fiqcm / 64.0;
        f3k -= 7.0 * f3qcm[(k, k, k)].powi(2) / (576.0 * freq[k]);
        let wk = freq[k].powi(2);

        // kkl terms
        for ll in 0..n1dm {
            let l = i1mode[ll];
            if k == l {
                continue;
            }
            let wl = freq[l].powi(2);
            let zval1 = f3qcm[(k, k, l)].powi(2);
            let res = ifrm1.get(&k);
            if res.is_some() && *res.unwrap() == l {
                let delta = 2.0 * (2.0 * freq[k] + freq[l]);
                f3kkl += 3.0 * zval1 / (64.0 * delta);
            } else {
                let znum1 = freq[l];
                let delta = 4.0 * wk - wl;
                f3kkl += 3.0 * zval1 * znum1 / (64.0 * delta);
            }
        }
    }
    // klm terms
    let mut f3klm = 0.0;
    for kk in 0..n1dm {
        let k = i1mode[kk];
        for ll in 0..n1dm {
            let l = i1mode[ll];
            if k <= l {
                continue;
            }
            for mm in 0..n1dm {
                let m = i1mode[mm];
                if l <= m {
                    continue;
                }
                let zval3 = f3qcm[(k, l, m)].powi(2);
                let xklm = freq[k] * freq[l] * freq[m];
                let d1 = freq[k] + freq[l] + freq[m];
                let d2 = freq[k] - freq[l] + freq[m];
                let d3 = freq[k] + freq[l] - freq[m];
                let d4 = freq[k] - freq[l] - freq[m];
                if ifrmchk[(k, l, m)] != 0 {
                    let delta1 = 1.0 / d1 + 1.0 / d2 + 1.0 / d4;
                    f3klm -= zval3 * delta1 / 16.0;
                } else if ifrmchk[(l, m, k)] != 0 {
                    let delta1 = 1.0 / d1 + 1.0 / d2 - 1.0 / d3;
                    f3klm -= zval3 * delta1 / 16.0;
                } else if ifrmchk[(k, m, l)] != 0 {
                    let delta1 = 1.0 / d1 - 1.0 / d3 + 1.0 / d4;
                    f3klm -= zval3 * delta1 / 16.0;
                } else {
                    let delta2 = d1 * d2 * d3 * d4;
                    f3klm -= zval3 * xklm / (4.0 * delta2);
                }
            }
        }
    }
    // biggest differences in f4k and f3klm, but I think it's okay
    let e0 = f4k + f3k + f3kkl + f3klm;
    e0
}

/// compute the fundamental frequencies from the harmonic frequencies and the
/// anharmonic constants
pub fn make_funds(freq: &Dvec, nvib: usize, xcnst: &Dmat) -> Vec<f64> {
    let mut fund = Vec::with_capacity(freq.len());
    for i in 0..nvib {
        let mut val = freq[i] + 2.0 * xcnst[(i, i)];
        for j in 0..nvib {
            if j != i {
                val += 0.5 * xcnst[(i, j)];
            }
        }
        fund.push(val);
    }
    fund
}

/// take a vec of energy: state pairs and print them in SPECTRO's format
#[allow(dead_code)]
pub(crate) fn print_vib_states(reng: &[f64], i1sts: &Vec<Vec<usize>>) {
    println!(
        "{:^10}{:^20}{:^20}{:>21}",
        "STATE NO.", "ENERGY (CM-1)", "ABOVE ZPT", "VIBRATIONAL STATE"
    );
    for (i, (energy, state)) in zip(reng, i1sts).enumerate() {
        print!("{:5}{:20.4}{:20.4}", i + 1, energy, *energy - reng[0]);
        print!("{:>21}", "NON-DEG (Vs) :");
        for s in state {
            print!("{:5}", s);
        }
        println!();
    }
}

pub(crate) fn rsfrm2(
    ijst: usize,
    kst: usize,
    ivib: usize,
    jvib: usize,
    kvib: usize,
    f3qcm: &F3qcm,
    eng: &mut [f64],
) {
    let val = f3qcm[(ivib, jvib, kvib)] / (2.0 * SQRT_2);
    let eres = dmatrix![
    eng[ijst] - eng[0], val;
    val, eng[kst] - eng[0];
    ];
    // TODO left out error measures
    let (eigval, eigvec) = symm_eigen_decomp(eres);
    let a = eigvec[(0, 0)];
    let b = eigvec[(1, 0)];
    if a.abs() > b.abs() {
        eng[ijst] = eigval[0] + eng[0];
        eng[kst] = eigval[1] + eng[0];
    } else {
        eng[kst] = eigval[0] + eng[0];
        eng[ijst] = eigval[1] + eng[0];
    }
    // TODO left out properties
}

/// calculate the type-1 fermi resonance contribution to the energy. `deg` is
/// usually false, but true in one case for symmetric tops
pub(crate) fn rsfrm1(
    ist: usize,
    jst: usize,
    ivib: usize,
    jvib: usize,
    f3qcm: &F3qcm,
    eng: &mut [f64],
    deg: bool,
) {
    let val = if deg {
        f3qcm[(ivib, ivib, jvib)] / f64::sqrt(8.0)
    } else {
        0.25 * f3qcm[(ivib, ivib, jvib)]
    };
    // this is actually a symmetric matrix I think
    let eres = [eng[ist] - eng[0], val, eng[jst] - eng[0]];
    // TODO left out printed error measures
    let eres = dmatrix![
    eres[0], val;
    val, eres[2];
    ];
    let (eigval, eigvec) = symm_eigen_decomp(eres);
    let a = eigvec[(0, 0)];
    let b = eigvec[(1, 0)];
    if a.abs() > b.abs() {
        eng[ist] = eigval[0] + eng[0];
        eng[jst] = eigval[1] + eng[0];
    } else {
        eng[ist] = eigval[1] + eng[0];
        eng[jst] = eigval[0] + eng[0];
    }
    // TODO left out the calculation of updated properties
}

/// convert tau to tau prime in wavenumbers
pub(crate) fn tau_prime(maxcor: usize, tau: &Tensor4) -> Dmat {
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
    taupcm
}

pub(crate) fn make_tau(
    maxcor: usize,
    nvib: usize,
    freq: &Dvec,
    primat: &[f64],
    wila: &Dmat,
) -> Tensor4 {
    // convert to cm-1 from the biggest mess you've ever seen
    const CONST1: f64 = 3.833384078e04;
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
                }
            }
        }
    }
    tau
}

/// set up vectors for principal -> cartesian and cartesian -> principal
/// transformations
pub(crate) fn princ_cart(irep: usize) -> ([usize; 3], [usize; 3]) {
    let ic = [IPTOC[(irep, 0)], IPTOC[(irep, 1)], IPTOC[(irep, 2)]];
    let id = [ICTOP[(irep, 0)], ICTOP[(irep, 1)], ICTOP[(irep, 2)]];
    (ic, id)
}

#[cfg(test)]
mod tests {
    use super::*;

    use approx::assert_abs_diff_eq;
    use nalgebra::dmatrix;

    use crate::FACT2;

    #[test]
    fn test_find3r() {
        let got = find3(2, 2, 2);
        let want = 9;
        assert_eq!(got, want);
    }

    #[test]
    fn test_taupcm() {
        let s = Spectro::load("testfiles/h2o/spectro.in");
        let fc2 = load_fc2("testfiles/fort.15", s.n3n);
        let fc2 = s.rot2nd(fc2);
        let fc2 = FACT2 * fc2;
        let w = s.geom.weights();
        let sqm: Vec<_> = w.iter().map(|w| 1.0 / w.sqrt()).collect();
        let fxm = s.form_sec(fc2, &sqm);
        let (harms, lxm) = symm_eigen_decomp(fxm);
        let freq = to_wavenumbers(&harms);
        let (_zmat, wila) = s.zeta(&lxm, &w);
        let tau = make_tau(3, 3, &freq, &s.primat, &wila);
        let got = tau_prime(3, &tau);
        let want = dmatrix![
        -0.08628870,  0.01018052, -0.00283749;
         0.01018052, -0.00839612, -0.00138895;
        -0.00283749, -0.00138895, -0.00093412;
           ];
        assert_abs_diff_eq!(got, want, epsilon = 2e-7);
    }
}

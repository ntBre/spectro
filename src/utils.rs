use std::{
    fmt::{Debug, Display},
    fs::read_to_string,
    str::FromStr,
};

use nalgebra::SymmetricEigen;
use tensor::{Tensor3, Tensor4};

use crate::{Dmat, Dvec, Spectro, FACT3, FACT4, FUNIT3, FUNIT4, WAVE};

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
        .map(|s| s.parse::<T>().unwrap())
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
pub(crate) fn find3r(i: usize, j: usize, k: usize) -> usize {
    let [i, j, k] = sort_indices([i + 1, j + 1, k + 1]);
    i + (j - 1) * j / 2 + (k - 1) * k * (k + 1) / 6 - 1
}

/// quartic force constant indexing formula. it relies on the fortran numbering,
/// so I need to add one initially and then subtract one at the end
pub(crate) fn find4t(i: usize, j: usize, k: usize, l: usize) -> usize {
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

pub fn load_fc2(infile: &str, n3n: usize) -> Dmat {
    let data = read_to_string(infile).unwrap();
    Dmat::from_iterator(
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

pub fn load_fc4(infile: &str, n3n: usize) -> Tensor4 {
    let mut f4x = Tensor4::zeros(n3n, n3n, n3n, n3n);
    let f44 = load_fc34("testfiles/fort.40");
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

fn load_fc34(infile: &str) -> Vec<f64> {
    let data = read_to_string(infile).unwrap();
    data.split_whitespace()
        .map(|s| s.parse().unwrap())
        .collect()
}

/// freq is the vector of harmonic frequencies
pub fn force3(
    n3n: usize,
    f3x: &mut Tensor3,
    lx: &Dmat,
    nvib: usize,
    freq: &Dvec,
    i3vib: usize,
) -> Vec<f64> {
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
        let wk = freq[ivib];
        for ii in 0..nvib {
            let wi = freq[ii];
            for jj in 0..nvib {
                let wj = freq[jj];
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
    let mut f3qcm =
        Vec::with_capacity(find3r(nvib - 1, nvib - 1, nvib - 1) + 1);
    for i in 0..nvib {
        let wi = freq[(i)];
        for j in 0..=i {
            let wj = freq[(j)];
            for k in 0..=j {
                let wk = freq[(k)];
                let wijk = wi * wj * wk;
                let fact = FACT3 / wijk.sqrt();
                f3qcm.push(f3q[(i, j, k)] * fact);
            }
        }
    }
    f3qcm
}

pub fn force4(
    n3n: usize,
    f4x: &mut Tensor4,
    lx: &Dmat,
    nvib: usize,
    harms: &Dvec,
    i4vib: usize,
) -> Vec<f64> {
    let mut f4q = Tensor4::zeros(n3n, n3n, n3n, n3n);
    for kabc in 0..n3n {
        for labc in 0..n3n {
            let mut dd = Dmat::zeros(n3n, n3n);
            for i in 0..n3n {
                for j in 0..n3n {
                    dd[(i, j)] = f4x[(i, j, kabc, labc)];
                }
            }
            dd *= FUNIT4;
            let ee = lx.clone().transpose() * dd * lx.clone();
            for i in 0..n3n {
                for j in 0..n3n {
                    f4q[(i, j, kabc, labc)] = ee[(i, j)];
                }
            }
        }
    }
    for i in 0..n3n {
        for j in 0..n3n {
            let mut dd = Dmat::zeros(n3n, n3n);
            for kabc in 0..n3n {
                for labc in 0..n3n {
                    dd[(kabc, labc)] = f4q[(i, j, kabc, labc)];
                }
            }
            let ee = lx.clone().transpose() * dd * lx.clone();
            for kabc in 0..n3n {
                for labc in 0..n3n {
                    f4x[(kabc, labc, i, j)] = ee[(kabc, labc)];
                }
            }
        }
    }
    let mut frq4 = Tensor4::zeros(nvib, nvib, nvib, nvib);
    let n = nvib - 1;
    let mut f4qcm = vec![0.0; find4t(n, n, n, n) + 1];
    for ivib in 0..nvib {
        let wk = harms[ivib];
        for jvib in 0..=ivib {
            let wl = harms[jvib];
            let mut dd = Dmat::zeros(nvib, nvib);
            for ii in 0..nvib {
                let wi = harms[ii];
                for jj in 0..=ii {
                    let wj = harms[jj];
                    let wijkl = wi * wj * wk * wl;
                    let sqws = wijkl.sqrt();
                    let fact = FACT4 / sqws;
                    dd[(ii, jj)] = f4x[(ii, jj, ivib, jvib)] * fact;
                    dd[(jj, ii)] = f4x[(jj, ii, ivib, jvib)] * fact;
                    frq4[(ii, jj, ivib, jvib)] = f4x[(ii, jj, ivib, jvib)];
                    frq4[(jj, ii, ivib, jvib)] = f4x[(jj, ii, ivib, jvib)];
                    frq4[(jj, ii, jvib, ivib)] = f4x[(jj, ii, jvib, ivib)];
                    frq4[(ii, jj, jvib, ivib)] = f4x[(ii, jj, jvib, ivib)];
                    let ijkl = find4t(ivib, jvib, ii, jj);
                    f4qcm[ijkl] = dd[(ii, jj)];
                }
            }
        }
    }
    let facts4 = quartic_sum_facs(i4vib, nvib);
    f4qcm
}

fn quartic_sum_facs(i4vib: usize, nvib: usize) -> Vec<f64> {
    let mut facts4 = vec![1.0; i4vib];
    for i in 0..nvib {
        facts4[find4t(i, i, i, i)] = 24.0;
        // intentionally i-1
        for j in 0..i {
            facts4[find4t(i, i, i, j)] = 6.0;
            facts4[find4t(i, j, j, j)] = 6.0;
            facts4[find4t(i, i, j, j)] = 4.0;
            for k in 0..j {
                facts4[find4t(i, i, j, k)] = 2.0;
                facts4[find4t(i, j, j, k)] = 2.0;
                facts4[find4t(i, j, k, k)] = 2.0;
            }
        }
    }
    facts4
}

/// calculate the anharmonic constants
pub fn xcalc(
    nvib: usize,
    f4qcm: &[f64],
    freq: &Dvec,
    f3qcm: &[f64],
    zmat: &Tensor3,
) -> Dmat {
    // TODO filled by loading fermi resonances
    let ifrmchk = Tensor3::zeros(30, 30, 30);
    let mut xcnst = Dmat::zeros(nvib, nvib);
    // diagonal contributions to the anharmonic constants
    for k in 0..nvib {
        let kkkk = find4t(k, k, k, k);
        let val1 = f4qcm[kkkk] / 16.0;
        let wk = freq[k].powi(2);
        let mut valu = 0.0;
        let mut ifrmck = false; // probably a bool
        for l in 0..nvib {
            let kkl = find3r(k, k, l);
            let val2 = f3qcm[kkl].powi(2);
            if ifrmchk[(k, k, l)] != 0.0 {
                ifrmck = true;
                todo!();
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
    for k in 1..nvib {
        for l in 0..k {
            let kkll = find4t(k, k, l, l);
            let val1 = f4qcm[kkll] / 4.0;
            let mut val2 = 0.0;
            for m in 0..nvib {
                let kkm = find3r(k, k, m);
                let llm = find3r(l, l, m);
                val2 -= f3qcm[kkm] * f3qcm[llm] / (4.0 * freq[m]);
            }

            let mut valu = 0.0;
            let mut ifrmck = false;
            for m in 0..nvib {
                let lm = ioff(l.max(m)) + l.min(m);
                let km = ioff(k.max(m)) + k.min(m);
                let d1 = freq[k] + freq[l] + freq[m];
                let d2 = freq[k] - freq[l] + freq[m];
                let d3 = freq[k] + freq[l] - freq[m];
                let d4 = -freq[k] + freq[l] + freq[m];
                let klm = find3r(k, l, m);
                // TODO fermi check stuff
                // if ifrmchk[(l, m, k)] != 0.0 && m != l {
                // 	ifrmck = true;
                // 	todo!();
                // }
                let delta = -d1 * d2 * d3 * d4;
                let val3 = freq[m].powi(2) - freq[k].powi(2) - freq[l].powi(2);
                valu -= 0.5 * f3qcm[klm].powi(2) * freq[m] * val3 / delta;
                valu;
            }
            let val5 = freq[k] / freq[l];
            let val6 = freq[l] / freq[k];
            // TODO actually compute these rotational constants
            let rotcon = vec![
                27.280988597962384,
                14.576838043259453,
                9.5005064610993717,
            ];
            let val7 = rotcon[0] * zmat[(k, l, 0)].powi(2)
                + rotcon[1] * zmat[(k, l, 1)].powi(2)
                + rotcon[2] * zmat[(k, l, 2)].powi(2);
            let val8 = (val5 + val6) * val7;
            let value = val1 + val2 + valu + val8;
            xcnst[(k, l)] = value;
            xcnst[(l, k)] = value;
        }
    }
    xcnst
}

/// compute the fundamental frequencies from the harmonic frequencies and the
/// anharmonic constants
pub fn funds(freq: &Dvec, nvib: usize, xcnst: &Dmat) -> Vec<f64> {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_find3r() {
        let got = find3r(2, 2, 2);
        let want = 9;
        assert_eq!(got, want);
    }

    #[test]
    fn test_quartic_sum_facs() {
        let got = quartic_sum_facs(15, 3);
        let want: Vec<f64> = vec![
            24.0, 6.0, 4.0, 6.0, 24.0, 6.0, 2.0, 2.0, 6.0, 4.0, 2.0, 4.0, 6.0,
            6.0, 24.0,
        ];
        assert_eq!(got, want);
    }
}

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

/// cubic force constant indexing formula. I think it relies on the fortran
/// numbering though, so I need to add one initially and then subtract one at
/// the end
fn find3r(i: usize, j: usize, k: usize) -> usize {
    let (i, j, k) = (i + 1, j + 1, k + 1);
    (i - 1) * i * (i + 1) / 6 + (j - 1) * j / 2 + k - 1
}

/// quartic force constant indexing formula. it relies on the fortran numbering,
/// so I need to add one initially and then subtract one at the end
fn find4t(i: usize, j: usize, k: usize, l: usize) -> usize {
    let (i, j, k, l) = (i + 1, j + 1, k + 1, l + 1);
    (i - 1) * i * (i + 1) * (i + 2) / 24 + find3r(j - 1, k - 1, l - 1)
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

pub fn force3(
    n3n: usize,
    f3x: &mut Tensor3,
    lx: &Dmat,
    nvib: usize,
    harms: &Dvec,
    i3vib: usize,
) {
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
        let wk = harms[ivib];
        for ii in 0..nvib {
            let wi = harms[ii];
            for jj in 0..nvib {
                let wj = harms[jj];
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
}

pub fn force4(
    n3n: usize,
    f4x: &mut Tensor4,
    lx: &Dmat,
    nvib: usize,
    harms: &Dvec,
    i4vib: usize,
) {
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
                }
            }
        }
    }
    let facts4 = quartic_sum_facs(i4vib, nvib);
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

#[cfg(test)]
mod tests {
    use super::*;

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

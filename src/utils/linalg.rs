use crate::Dvec;

use crate::Dmat;
use crate::Mat3;

use nalgebra::Vector3;
use nalgebra_lapack::SymmetricEigen;

/// compute the eigen decomposition of the symmetric matrix `mat` and return
/// both the sorted eigenvalues and the corresponding eigenvectors in descending
/// order. the implementation is taken from RSP and the subroutines called
/// therein
pub fn symm_eigen_decomp(mat: Dmat) -> (Dvec, Dmat) {
    // let (n, m, a, mut d, e) = tred3(mat);
    // let mut z = Dmat::identity(n, n);
    // tql2(n, e, &mut d, m, &mut z);
    // trbak3(n, a, m, &mut z);
    // let w = Dvec::from(d);
    // (w, z)
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

/// copy of the above with constant-size matrices because I can't figure out how
/// to make it generic
#[allow(unused)]
pub fn symm_eigen_decomp3(mat: Mat3, reverse: bool) -> (Vector3<f64>, Mat3) {
    let SymmetricEigen {
        eigenvectors: vecs,
        eigenvalues: vals,
    } = SymmetricEigen::new(mat);
    let mut pairs: Vec<_> = vals.iter().enumerate().collect();
    pairs.sort_by(|(_, a), (_, b)| b.partial_cmp(&a).unwrap());
    if reverse {
        pairs.reverse();
    }
    let (_, cols) = vecs.shape();
    let mut ret = Mat3::zeros();
    for i in 0..cols {
        ret.set_column(i, &vecs.column(pairs[i].0));
    }
    (
        Vector3::from_iterator(pairs.iter().map(|a| a.1.clone())),
        ret,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use nalgebra::{dmatrix, dvector};

    #[test]
    #[ignore]
    fn test_tred3() {
        let inp = dmatrix![
            3.7432958001669672, 0.0, 0.0;
            8.4648439005796661e-05, 4.2568667187161822, 0.0;
            0.0, 0.0, 3.7432957395278987;
        ];

        let (n, m, a, d, e) = tred3(inp);
        assert_eq!(n, 3);
        assert_eq!(m, 3);
        assert_abs_diff_eq!(
            a,
            dmatrix![
            0.0, 0.0, 0.0;
            0.00016929687801159332, 0.00011971097047570936, 0.0;
            0.0, 0.0, 0.0;
            ]
        );

        assert_abs_diff_eq!(
            Dvec::from(d),
            dvector![
                3.7432958001669672,
                4.2568667187161822,
                3.7432957395278987
            ],
        );

        assert_abs_diff_eq!(
            Dvec::from(e),
            dvector![0.0, -8.4648439005796661e-05, 0.0],
        );
    }
}

/// THIS SUBROUTINE IS A TRANSLATION OF THE ALGOL PROCEDURE TRBAK3, NUM. MATH.
/// 11, 181-195(1968) BY MARTIN, REINSCH, AND WILKINSON. HANDBOOK FOR AUTO.
/// COMP., VOL.II-LINEAR ALGEBRA, 212-226(1971), by way of the Fortran
/// implementation in spectro.
///
/// THIS SUBROUTINE FORMS THE EIGENVECTORS OF A REAL SYMMETRIC MATRIX BY BACK
/// TRANSFORMING THOSE OF THE CORRESPONDING SYMMETRIC TRIDIAGONAL MATRIX
/// DETERMINED BY TRED3.
#[allow(unused)]
fn trbak3(n: usize, a: Dmat, m: usize, z: &mut Dmat) {
    for i in 1..n {
        let l = i - 1;
        let iz = i * l / 2;
        let ik = iz + 1;
        let h = a[ik];
        if h == 0.0 {
            continue;
        }

        // m is the number of eigenvectors, so it should be set correctly from
        // waaaaay above
        for j in 0..m {
            let mut s = 0.0;
            let mut ik = iz;

            for k in 0..=l {
                s += a[ik] * z[(k, j)];
                ik += 1;
            }

            // supposedly avoiding possible underflow
            s = (s / h) / h;
            ik = iz;
            for k in 0..=l {
                z[(k, j)] -= s * a[ik];
                ik += 1;
            }
        }
    }
}

/// THIS SUBROUTINE IS A TRANSLATION OF THE ALGOL PROCEDURE TQL2, NUM. MATH. 11,
/// 293-306(1968) BY BOWDLER, MARTIN, REINSCH, AND WILKINSON. HANDBOOK FOR AUTO.
/// COMP., VOL.II-LINEAR ALGEBRA, 227-240(1971), by way of the Fortran
/// implementation in spectro
///
/// THIS SUBROUTINE FINDS THE EIGENVALUES AND EIGENVECTORS OF A SYMMETRIC
/// TRIDIAGONAL MATRIX BY THE QL METHOD. THE EIGENVECTORS OF A FULL SYMMETRIC
/// MATRIX CAN ALSO BE FOUND IF TRED2 HAS BEEN USED TO REDUCE THIS FULL MATRIX
/// TO TRIDIAGONAL FORM.
#[allow(unused)]
fn tql2(n: usize, mut e: Vec<f64>, d: &mut Vec<f64>, m: usize, z: &mut Dmat) {
    // this is supposed to be a machine dependent parameter specifying the
    // precision of floats. in the fortran code it comes out to about 7.1e-15
    let machep = 2.0_f64.powf(-47.0);
    // moving them all over one? lol
    for i in 1..n {
        e[i - 1] = e[i];
    }
    let mut f = 0.0;
    let mut b = 0.0;
    e[n] = 0.0;
    for l in 0..n {
        let h = machep * (d[l].abs() + e[l].abs());
        if b < h {
            b = h;
        }
        // look for small sub-diagonal element
        for m in l..n {
            if e[m].abs() <= b {
                // goto 120, should break and keep m
                break;
            }
        }

        if m == l {
            // goto 220
        }
        // form shift
        let l1 = l + 1;
        let g = d[l];
        let p = (d[l1] - g) / (2.0 * e[l]);
        let r = (p * p + 1.0).sqrt();
        d[l] = e[l] / (p + r.copysign(p));
        let h = g - d[l];

        for i in l1..n {
            d[i] -= h;
        }

        f += h;
        // QL transformation
        let mut p = d[m];
        let mut c = 1.0;
        let mut s = 0.0;
        let mml = m - l;

        // TODO another reverse step
        for ii in 0..mml {
            let i = m - ii;
            let g = c * e[i];
            let h = c * p;
            if p.abs() < e[i].abs() {
                // goto 150:
                c = p / e[i];
                let r = (c * c + 1.0).sqrt();
                e[i + 1] = s * e[i] * r;
                s = 1.0 / r;
                c = c * s;
            } else {
                c = e[i] / p;
                let r = (c * c + 1.0).sqrt();
                e[i + 1] = s * p * r;
                s = c / r;
                c = 1.0 / r;
            };
            // this is 160
            p = c * d[i] - s * g;
            d[i + 1] = h + s * (c * g + s * d[i]);
            // form vector
            for k in 0..n {
                let h = z[(k, i + 1)];
                z[(k, i + 1)] = s * z[(k, i)] + c * h;
                z[(k, i)] = c * z[(k, i)] - s * h;
            }
        }
        e[l] = s * p;
        d[l] = c * p;
        if e[l].abs() > b {
            todo!("goto 130, after the m == l thing");
        }
        d[l] += f;
    }
    // order eigenvalues and eigenvectors
    for ii in 1..n {
        let i = ii - 1;
        let mut k = i;
        let mut p = d[i];

        for j in ii..n {
            if d[j] >= p {
                continue;
            }
            k = j;
            p = d[j];
        }

        if k == i {
            continue;
        }
        d[k] = d[i];
        d[i] = p;

        for j in 0..n {
            let p = z[(j, i)];
            z[(j, i)] = z[(j, k)];
            z[(j, k)] = p;
        }
    }
}

/// THIS SUBROUTINE IS A TRANSLATION OF THE ALGOL PROCEDURE TRED3, NUM. MATH.
/// 11, 181-195(1968) BY MARTIN, REINSCH, AND WILKINSON. HANDBOOK FOR AUTO.
/// COMP., VOL.II-LINEAR ALGEBRA, 212-226(1971), by way of the fortran version
/// in spectro
///
/// THIS SUBROUTINE REDUCES A REAL SYMMETRIC MATRIX, `mat` ARRAY, TO A SYMMETRIC
/// TRIDIAGONAL MATRIX, represented as two vectors, USING ORTHOGONAL SIMILARITY
/// TRANSFORMATIONS.
#[allow(unused)]
fn tred3(mat: Dmat) -> (usize, usize, Dmat, Vec<f64>, Vec<f64>) {
    // N is the order of the matrix
    let (n, m) = mat.shape();
    assert_eq!(n, m, "non-square matrix passed to symm_eigen_decomp");
    // A contains the lower triangle of the real symmetric input matrix, stored
    // row-wise
    let mut a = mat.lower_triangle();
    // on output, w contains the eigenvalues in ascending order and z contains
    // the eigenvectors
    // D contains the diagonal elements of the tridiagonal matrix
    let mut d = vec![0.0; n];
    // E contains the subdiagonal elements of the tridiagonal matrix in its last
    // n-1 positions. E(1) is set to zero
    let mut e = vec![0.0; n];
    // E2 contains the squares of the corresponding elements of E
    let mut e2 = vec![0.0; n];

    // using fortran indexing so the math works out
    for ii in 1..=n {
        // I think this +1 is fortran only
        let i = n + 1 - ii;
        let l = i - 1;
        let mut iz = (i * l) / 2;
        let mut h = 0.0;
        let mut scale = 0.0;
        if l < 1 {
            e[i] = 0.0;
            e2[i] = 0.0;
        } else {
            for k in 0..l {
                iz += 1;
                d[k] = a[iz];
                scale += d[k].abs();
            }

            if scale != 0.0 {
                for k in 0..=l {
                    d[k] /= scale;
                    h += d[k] * d[k];
                }
                e2[i - 1] = scale * scale * h;
                let f = d[l];
                let g = -(h.sqrt().copysign(f));
                e[i - 1] = scale * g;
                h -= f * g;
                a[iz] = scale * d[l];
                // goto 290 if l == 0
                if l != 0 {
                    let mut f = 0.0;
                    for j in 0..=l {
                        let mut g = 0.0;
                        let mut jk = if j >= 1 { (j * (j - 1)) / 2 } else { 0 };
                        // form element of A*U
                        for k in 0..=l {
                            if k > j {
                                jk = jk + k - 2;
                            }
                            g += a[jk] * d[k];
                            jk += 1;
                        }
                        // form element of P
                        e[j] = g / h;
                        f += e[j] * d[j];
                    }

                    let hh = f / (h + h);
                    let mut jk = 0;
                    // form reduced A
                    for j in 0..=l {
                        let f = d[j];
                        let g = e[j] - hh * f;
                        e[j] = g;

                        for k in 0..=j {
                            a[jk] -= f * e[k] + g * d[k];
                            jk += 1;
                        }
                    }
                }
            } else {
                e[i] = 0.0;
                e2[i] = 0.0;
            }
        }
        // this is 290
        d[i - 1] = a[iz + 1];
        a[iz + 1] = scale * h.sqrt();
    }
    (n, m, a, d, e)
}

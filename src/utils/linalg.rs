use std::ops::IndexMut;

use crate::Dvec;

use crate::Dmat;
use crate::Mat3;

use nalgebra::vector;
use nalgebra::Dim;
use nalgebra::Matrix;
use nalgebra::Storage;
use nalgebra::SymmetricEigen;
use nalgebra::Vector3;

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

/// compute the eigen decomposition of the symmetric 3x3 matrix `mat` and return
/// the eigenvalues and eigenvectors sorted in ascending order
pub fn symm_eigen_decomp3(mat: Mat3) -> (Vector3<f64>, Mat3) {
    let (n, m, a, mut d, e) = tred3(mat);
    let mut z = Mat3::identity();
    tql2(n, e, &mut d, m, &mut z);
    trbak3(n, a, m, &mut z);
    let w = vector![d[0], d[1], d[2]];
    (w, z)
}

/// THIS SUBROUTINE IS A TRANSLATION OF THE ALGOL PROCEDURE TRED3, NUM. MATH.
/// 11, 181-195(1968) BY MARTIN, REINSCH, AND WILKINSON. HANDBOOK FOR AUTO.
/// COMP., VOL.II-LINEAR ALGEBRA, 212-226(1971), by way of the fortran version
/// in spectro
///
/// THIS SUBROUTINE REDUCES A REAL SYMMETRIC MATRIX, `mat` ARRAY, TO A SYMMETRIC
/// TRIDIAGONAL MATRIX, represented as two vectors, USING ORTHOGONAL SIMILARITY
/// TRANSFORMATIONS.
fn tred3<D: Dim, S: Storage<f64, D, D>>(
    mat: Matrix<f64, D, D, S>,
) -> (usize, usize, Vec<f64>, Vec<f64>, Vec<f64>) {
    let (n, _) = mat.shape();
    // a contains the lower triangle of mat row-wise
    let mut a = vec![0.0; n * (n / 2 + 1)];
    let mut ij = 0;
    for i in 0..n {
        for j in 0..=i {
            a[ij] = mat[(i, j)];
            ij += 1;
        }
    }
    let mut d = vec![0.0; n];
    let mut e = vec![0.0; n];
    let mut e2 = vec![0.0; n];
    for i in (0..n).rev() {
        // this covers l < 1, so else is goto 130
        if i >= 1 {
            let l = i - 1;
            let mut iz = (i * i) / 2;
            let mut h = 0.0;
            let mut scale = 0.0;
            for k in 0..i {
                iz += 1;
                d[k] = a[iz];
                scale += d[k].abs();
            }

            if scale != 0.0 {
                for k in 0..i {
                    d[k] /= scale;
                    h += d[k] * d[k];
                }

                e2[i] = scale * scale * h;
                let f = d[l];
                let g = -h.sqrt().copysign(f);
                e[i] = scale * g;
                h -= f * g;
                d[l] = f - g;
                a[iz] = scale * d[l];
                if l != 0 {
                    let mut f = 0.0;
                    for j in 0..i {
                        let mut g = 0.0;
                        let mut jk = (j * (j + 1)) / 2;
                        for k in 0..i {
                            if k > j {
                                jk = jk + k - 1;
                            }
                            g += a[jk] * d[k];
                            jk += 1;
                        }
                        e[j] = g / h;
                        f += e[j] * d[j];
                    }

                    let hh = f / (h + h);
                    let mut jk = 0;

                    for j in 0..i {
                        let f = d[j];
                        let g = e[j] - hh * f;
                        e[j] = g;

                        for k in 0..i {
                            jk += 1;
                            a[jk] = a[jk] - f * e[k] - g * d[k];
                        }
                    }
                }
            } else {
                e[i] = 0.0;
                e2[i] = 0.0;
            }
            // this is 290
            d[i] = a[iz + 1];
            a[iz + 1] = scale * h.sqrt();
        } else {
            e[i] = 0.0;
            e2[i] = 0.0;
            // copy of 290 with iz = 0
            d[i] = a[0];
            a[0] = 0.0;
        }
    }
    (n, n, a, d, e)
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
///
/// on output, d contains the eigenvalues in ascending order and z contains the
/// eigenvectors of the tridiagonal matrix build by tred3
fn tql2<D: Dim, S: Storage<f64, D, D>>(
    n: usize,
    mut e: Vec<f64>,
    d: &mut Vec<f64>,
    m: usize,
    z: &mut Matrix<f64, D, D, S>,
) where
    Matrix<f64, D, D, S>: IndexMut<(usize, usize), Output = f64>,
{
    let machep = 2.0_f64.powf(-47.0);

    if n == 1 {
        return;
    }

    // moving them all over one? lol
    for i in 1..n {
        e[i - 1] = e[i];
    }

    let mut f = 0.0;
    let mut b = 0.0;
    e[n - 1] = 0.0;

    // this is for skipping the top of the loop when we goto 130
    let mut skip = false;

    'outer: for l in 0..n {
        if !skip {
            let h = machep * (d[l].abs() + e[l].abs());
            if b < h {
                b = h;
            }
            // look for small sub-diagonal element
            for m in l..n {
                if e[m].abs() <= b {
                    // goto 120
                    if m == l {
                        // goto 220
                        d[l] += f;
                        continue 'outer;
                    }
                    // else just break out of the m loop and keep moving
                    break;
                }
            }

            if m == l {
                d[l] += f;
                continue 'outer;
            }
        }
        skip = false;

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
        let mut p = d[m - 1];
        let mut c = 1.0;
        let mut s = 0.0;
        let mml = m - l - 1;

        for ii in 0..mml {
            let i = m - ii - 2;
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
            // goto 130
            skip = true;
            continue;
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

/// THIS SUBROUTINE IS A TRANSLATION OF THE ALGOL PROCEDURE TRBAK3, NUM. MATH.
/// 11, 181-195(1968) BY MARTIN, REINSCH, AND WILKINSON. HANDBOOK FOR AUTO.
/// COMP., VOL.II-LINEAR ALGEBRA, 212-226(1971), by way of the Fortran
/// implementation in spectro.
///
/// THIS SUBROUTINE FORMS THE EIGENVECTORS OF A REAL SYMMETRIC MATRIX BY BACK
/// TRANSFORMING THOSE OF THE CORRESPONDING SYMMETRIC TRIDIAGONAL MATRIX
/// DETERMINED BY TRED3.
fn trbak3<D: Dim, S: Storage<f64, D, D>>(
    n: usize,
    a: Vec<f64>,
    m: usize,
    z: &mut Matrix<f64, D, D, S>,
) where
    Matrix<f64, D, D, S>: IndexMut<(usize, usize), Output = f64>,
{
    if m == 0 || n == 1 {
        return;
    }

    for i in 1..n {
        let l = i - 1;
        let iz = i * i / 2;
        let ik = iz + i + 1;
        let h = a[ik];
        if h == 0.0 {
            continue;
        }

        for j in 0..m {
            let mut s = 0.0;
            let mut ik = iz;

            for k in 0..=l {
                ik += 1;
                s += a[ik] * z[(k, j)];
            }

            // supposedly avoiding possible underflow
            s = (s / h) / h;
            ik = iz;
            for k in 0..=l {
                ik += 1;
                z[(k, j)] -= s * a[ik];
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::{check_mat, check_vec};

    use super::*;
    use nalgebra::{dmatrix, dvector};

    #[test]
    fn test_tred3() {
        struct Test {
            inp: Dmat,
            n: usize,
            m: usize,
            a: Dvec,
            d: Dvec,
            e: Dvec,
        }

        let tests = [
            //
            Test {
                inp: dmatrix![
                159.1101420,0.0,0.0;
                0.0000000,144.3669747,0.0;
                0.0000000,-0.0068560,14.7431673;
                        ],
                n: 3,
                m: 3,
                a: dvector![
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    -0.013711901310795693,
                    0.0096957783998243459
                ],
                d: dvector![
                    159.1101420375779,
                    144.36697474228089,
                    14.743167295296997
                ],
                e: dvector![0.0, 0.0, 0.0068559506553978466],
            },
        ];

        for test in tests {
            let (n, m, a, d, e) = tred3(test.inp);
            assert_eq!(n, test.n);
            assert_eq!(m, test.m);
            check_vec!(Dvec::from(a), test.a, 1e-7, "tred3 a");
            check_vec!(Dvec::from(d), test.d, 1e-7, "tred3 d");
            check_vec!(Dvec::from(e), test.e, 1e-7, "tred3 e");
        }
    }

    #[test]
    fn test_tql2() {
        struct Test {
            inp: Dmat,
            d: Dvec,
            z: Dmat,
        }
        let tests = [
            //
            Test {
                inp: dmatrix![
                159.1101420,0.0,0.0;
                0.0000000,144.3669747,0.0;
                0.0000000,-0.0068560,14.7431673;
                        ],
                d: dvector![
                    14.743166932677951,
                    144.36697510489992,
                    159.1101420375779
                ],
                z: dmatrix![
                0.000000000000,0.000000000000,1.000000000000;
                -0.000052891138,0.999999998601,0.000000000000;
                0.999999998601,0.000052891138,0.000000000000;
                        ],
            },
        ];
        for test in tests {
            let (n, m, _a, mut d, e) = tred3(test.inp);
            let mut z = Dmat::identity(n, n);
            tql2(n, e, &mut d, m, &mut z);
            check_vec!(Dvec::from(d), test.d, 4.3e-8, "tql2");
            check_mat!(&z, &test.z, 4e-10, "tql2", "tql2");
        }
    }

    #[test]
    fn test_trbak3() {
        struct Test {
            inp: Dmat,
            z: Dmat,
        }
        let tests = [
            //
            Test {
                inp: dmatrix![
                159.1101420,0.0,0.0;
                0.0000000,144.3669747,0.0;
                0.0000000,-0.0068560,14.7431673;
                        ],
                z: dmatrix![
                0.000000000000,0.000000000000,1.000000000000;
                0.000052891138,-0.999999998601,0.000000000000;
                0.999999998601,0.000052891138,0.000000000000;
                        ],
            },
        ];
        for test in tests {
            let (n, m, a, mut d, e) = tred3(test.inp);
            let mut z = Dmat::identity(n, n);
            tql2(n, e, &mut d, m, &mut z);
            trbak3(n, a, m, &mut z);
            check_mat!(&z, &test.z, 4e-10, "trbak3", "c3hcn");
        }
    }
}

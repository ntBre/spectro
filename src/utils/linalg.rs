use std::ops::IndexMut;

use crate::Dvec;

use crate::Dmat;
use crate::Mat3;

use nalgebra::vector;
use nalgebra::Dim;
use nalgebra::Matrix;
use nalgebra::RawStorageMut;
use nalgebra::Storage;
use nalgebra::Vector3;

/// compute the eigen decomposition of the symmetric matrix `mat` and return
/// both the sorted eigenvalues and the corresponding eigenvectors in descending
/// order. the implementation is taken from RSP and the subroutines called
/// therein
pub fn symm_eigen_decomp(mat: Dmat, reverse: bool) -> (Dvec, Dmat) {
    let (n, m, a, mut d, e) = tred3(mat);
    let mut z = Dmat::identity(n, n);
    tql2(n, e, &mut d, &mut z);
    trbak3(n, a, m, &mut z);
    // reverse the eigenvalues and eigenvectors
    if reverse {
        d.reverse();
        let (r, c) = z.shape();
        let mut zret = Dmat::zeros(r, c);
        for i in 0..c {
            zret.set_column(i, &z.column(c - i - 1));
        }
        z = zret;
    }
    let w = Dvec::from(d);
    (w, z)
}

/// compute the eigen decomposition of the symmetric 3x3 matrix `mat` and return
/// the eigenvalues and eigenvectors sorted in ascending order
pub fn symm_eigen_decomp3(mat: Mat3) -> (Vector3<f64>, Mat3) {
    let (n, m, a, mut d, e) = tred3(mat);
    let mut z = Mat3::identity();
    tql2(n, e, &mut d, &mut z);
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
            let mut iz = ((i + 1) * i) / 2 - 1;
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

                    for j in 0..=l {
                        let f = d[j];
                        let g = e[j] - hh * f;
                        e[j] = g;

                        for k in 0..=j {
                            a[jk] = a[jk] - f * e[k] - g * d[k];
                            jk += 1;
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
    z: &mut Matrix<f64, D, D, S>,
) where
    Matrix<f64, D, D, S>: IndexMut<(usize, usize), Output = f64>,
    S: Clone,
    S: RawStorageMut<f64, D, D>,
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

    'outer: for l in 0..n {
        let h = machep * (d[l].abs() + e[l].abs());
        if b < h {
            b = h;
        }
        // look for small sub-diagonal element
        let mut m_outer = l;
        for m in l..n {
            if e[m].abs() <= b {
                // this is 120, only applicable in this case, l will never be
                // equal to the outer m
                if m == l {
                    // goto 220
                    d[l] += f;
                    continue 'outer;
                }
                // else just break out of the m loop and keep moving
                m_outer = m;
                break;
            }
        }

        // for keeping track of convergence. better name would be iter
        let mut j = 0;
        loop {
            if j == 30 {
                panic!("too many iterations looking for eigenvalue");
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
            let mut p = d[m_outer];
            let mut c = 1.0;
            let mut s = 0.0;
            let mml = m_outer - l;

            for ii in 0..mml {
                let i = m_outer - ii - 1;
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
                }
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
            if e[l].abs() <= b {
                break;
            }
            j += 1;
        }
        d[l] += f;
    }

    // BUG z actually looks okay here
    dbg!(&d);

    // order eigenvalues and eigenvectors. the fortran version is wacky, so just
    // sort into ascending order
    let mut ds: Vec<_> = d.iter().enumerate().collect();
    ds.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

    let vals: Vec<_> = ds.iter().map(|a| a.1.clone()).collect();
    let ids = ds.iter().map(|a| a.0);
    let mut zret = z.clone();
    for (i, id) in ids.enumerate() {
        zret.set_column(i, &z.column(id));
    }
    d.copy_from_slice(&vals);
    *z = zret;
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
        let iz = ((i + 1) * i) / 2 - 1;
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
    use crate::{check_mat, check_vec, tests::load_dmat};

    use super::*;
    use nalgebra::{dmatrix, dvector};

    #[test]
    fn test_tred3() {
        #[derive(Clone)]
        struct Test {
            label: &'static str,
            inp: Dmat,
            n: usize,
            m: usize,
            a: Dvec,
            d: Dvec,
            e: Dvec,
        }

        let tests = [
            Test {
                label: "c3hcn",
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
            Test {
                label: "c3hf",
                inp: dmatrix![
                        66.2593939, 0.0, 0.0;
                        0.0000000,81.3146042, 0.0;
                        -0.0035385,0.0000000,15.0552103;
                ],
                n: 3,
                m: 3,
                a: dvector![
                    0.0,
                    -2.8421709430404007e-14,
                    2.0097183471152322e-14,
                    -0.0035385188367391684,
                    0.0035385188367391684,
                    0.0035385188367391684
                ],
                d: dvector![
                    81.314604235348312,
                    66.25939388689622,
                    15.055210348452093
                ],
                e: dvector![
                    0.0,
                    1.4210854715202004e-14,
                    -0.0035385188367391684
                ],
            },
            Test {
                label: "h2o lxm",
                inp: load_dmat("testfiles/h2o/linalg_fxm", 9, 9),
                n: 9,
                m: 9,
                a: dvector![
                    0.0,
                    4.3065526730660941e-14,
                    3.0451925986620881e-14,
                    1.4295421431353462e-09,
                    -3.3955940063313309e-09,
                    2.6051534882644881e-09,
                    1.5216206566822212e-05,
                    1.5572006273398162e-05,
                    2.1772007754521747e-05,
                    2.1772007754521747e-05,
                    -0.002930178525032634,
                    0.0038232348215916367,
                    0.0,
                    0.0057610737193432397,
                    0.0053100395991602829,
                    -1.1828054001956041,
                    -0.70607522263440647,
                    0.0,
                    0.26832997253476026,
                    1.5577485141694851,
                    1.4825910161450215,
                    0.330873857898902,
                    -0.81541226344888862,
                    0.0,
                    0.010724693883152558,
                    0.40079199810508132,
                    -2.0243229916294818,
                    1.5863491873587336,
                    0.00017532098335837354,
                    -6.5181328653962645e-05,
                    0.0,
                    -0.00037429641281150339,
                    0.00032299402032887059,
                    -0.0012215477795125551,
                    0.0031874146338449591,
                    0.0024424577806283218,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    -0.00011934970742648719,
                    0.0,
                    0.00011939297051202227,
                    0.00011937134092920443
                ],
                d: dvector![
                    0.00050077372048364988,
                    2.9132788909123214e-09,
                    -9.3649939394372522e-07,
                    -0.00051262037947833003,
                    8.5853525287625558,
                    2.1079188794411872,
                    8.7356648193059137,
                    6.2442673095297607e-05,
                    0.00050077372048364978
                ],
                e: dvector![
                    0.0,
                    -2.1532763365330471e-14,
                    1.9987150067888276e-09,
                    -2.1772007754521747e-05,
                    -0.0048943169135257474,
                    -1.411059680789253,
                    1.2431335091481874,
                    -0.0018716109121189426,
                    -0.00011934971526486563
                ],
            },
            Test {
                label: "h2o sic lxm",
                inp: load_dmat("testfiles/h2o_sic/step_fxm", 9, 9),
                n: 9,
                m: 9,
                a: dvector![
                    0.0,
                    3.9020265119658704e-28,
                    2.7591494069807582e-28,
                    3.4741052077794205e-16,
                    -7.6798841469843432e-16,
                    5.9602863817865969e-16,
                    1.1130160404598044e-10,
                    1.266054456407815e-10,
                    1.6857338440308153e-10,
                    1.6857338440308153e-10,
                    -1.2013012806733059e-10,
                    -9.8065067177799392e-10,
                    0.0,
                    7.0599047913935285e-09,
                    5.0407520595278938e-09,
                    1.5291033414991011,
                    0.66895654399501081,
                    0.0,
                    -0.35292945347880272,
                    -1.76914041708843,
                    1.7378257825866976,
                    0.48362393184458341,
                    -1.0936377710178804,
                    0.0,
                    -0.085855445529061658,
                    0.61563836115016435,
                    1.3477082143017551,
                    1.3477082143017551,
                    -0.48379971112584819,
                    0.17984706055957517,
                    0.0,
                    1.0328643143238818,
                    -0.89146837038721116,
                    0.0,
                    -7.5439383920567007,
                    5.4331822229715891,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0
                ],
                d: dvector![
                    2.0066649276559488e-27,
                    -4.0509209018715553e-16,
                    2.3842744402326775e-11,
                    1.2191456723415539e-10,
                    8.4747414986738931,
                    1.6090725074973034,
                    5.9769349704040184,
                    3.3715917053301805,
                    0.0
                ],
                e: dvector![
                    0.0,
                    -1.9510132559829352e-28,
                    4.6257226115657424e-16,
                    -1.6857338440308153e-10,
                    -3.5990827180290173e-09,
                    1.7070654321454646,
                    -1.3477082143017551,
                    3.9130050557009666,
                    0.0
                ],
            },
        ];

        for test in Vec::from(&tests[..]) {
            let (n, m, a, d, e) = tred3(test.inp);
            assert_eq!(n, test.n);
            assert_eq!(m, test.m);
            check_vec!(Dvec::from(a), test.a, 1e-7, test.label);
            check_vec!(Dvec::from(d), test.d, 1e-7, test.label);
            check_vec!(Dvec::from(e), test.e, 1e-7, test.label);
        }
    }

    #[test]
    fn test_tql2() {
        #[derive(Clone)]
        struct Test {
            label: &'static str,
            inp: Dmat,
            d: Dvec,
            z: Dmat,
            eps: f64,
            zeps: f64,
        }
        let tests = [
            Test {
                label: "c3hcn",
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
                eps: 4.3e-8,
                zeps: 4e-10,
            },
            Test {
                label: "c3hf",
                inp: dmatrix![
                        66.2593939, 0.0, 0.0;
                        0.0000000,81.3146042, 0.0;
                        -0.0035385,0.0000000,15.0552103;
                ],
                d: dvector![
                    15.055210103919038,
                    66.259394131429275,
                    81.314604235348312
                ],
                z: dmatrix![
                    0.000000000000,0.000000000000,1.000000000000;
                    0.000069106049,0.999999997612,0.000000000000;
                    0.999999997612,-0.000069106049,0.000000000000;
                ],
                eps: 4.9e-8,
                zeps: 4e-10,
            },
            Test {
                label: "h2o lxm",
                inp: load_dmat("testfiles/h2o/linalg_fxm", 9, 9),
                d: dvector![
                    -0.00051671009837685373,
                    -1.7647455236096873e-08,
                    3.1072289771789239e-09,
                    3.1637048859384355e-05,
                    0.00050077372048364988,
                    0.00053113662711965816,
                    1.605890850741277,
                    8.659520126712037,
                    9.1635288634469454
                ],
                z: dmatrix![
                -0.000000000000,-0.000000000004,0.000000000043,0.000000000000,1.000000000000,-0.000000000000,-0.000000000000,0.000000000000,0.000000000000;
                -0.000000163138,-0.096668929314,0.995316591897,0.000000009685,-0.000000000043,0.000000000005,0.000000000000,-0.000000000000,-0.000000000000;
                0.042174761428,0.994430995516,0.096582923740,0.000153284915,-0.000000000000,0.000001216685,0.000000008068,-0.000000001078,-0.000000000789;
                0.999109899868,-0.041977206332,-0.004076822018,-0.000229332711,0.000000000000,-0.000029733830,-0.000595126801,0.000428562791,0.000332171058;
                0.000647250569,-0.000027200670,-0.000002641727,0.000024820361,0.000000000000,0.000006335594,0.195331396745,-0.758301488821,-0.621951828650;
                0.000472870337,-0.000019898228,-0.000001932515,0.000151810154,0.000000000000,0.000038648591,0.966161055692,0.039856076388,0.254841242921;
                -0.000067335863,0.000002865406,0.000000278290,-0.000229239800,-0.000000000000,-0.000058326531,-0.168457909239,-0.650684364175,0.740426721923;
                -0.000222995357,0.000157655964,0.000015321541,-0.969130125139,-0.000000000000,-0.246549476846,0.000196339572,0.000140635592,-0.000151229968;
                -0.000026157106,0.000037572921,0.000003651615,-0.246549484374,0.000000000000,0.969130202628,-0.000000014596,-0.000000001938,0.000000001970;
                    ],
                eps: 4.9e-8,
                zeps: 4.1e-6,
            },
            Test {
                label: "h2o sic lxm",
                inp: load_dmat("testfiles/h2o_sic/step_fxm", 9, 9),
                d: dvector![
                    -2.385447812605463e-10,
                    -4.050937453492381e-16,
                    0.0,
                    2.0066649276560428e-27,
                    8.6444250709664496e-11,
                    7.2971624774702655e-10,
                    1.6066663041276248,
                    8.6608186098499562,
                    9.1648557674959541
                ],
                z: dmatrix![
                -0.000000000000,0.000000000000,0.000000000000,1.000000000000,-0.000000000000,0.000000000000,0.000000000000,0.000000000000,0.000000000000;
                -0.000000931462,0.999999999989,0.000000000000,-0.000000000000,0.000004634218,-0.000000087955,0.000000000000,0.000000000000,-0.000000000000;
                0.480346766173,-0.000003578170,0.000000000000,0.000000000000,0.866034126102,-0.138751132034,0.000000000000,0.000000000000,-0.000000000000;
                0.747668440556,0.000002237942,0.000000000000,-0.000000000000,-0.321610917390,0.580997694321,-0.000000000435,-0.000000000313,0.000000000243;
                0.052382902446,0.000000243402,0.000000000000,-0.000000000000,-0.043732794441,-0.091618240785,0.193999171097,0.753339528760,-0.617894366538;
                -0.260055382845,-0.000001208370,0.000000000000,0.000000000000,0.217111845721,0.454839571243,-0.780521279197,0.082117088547,-0.249795767018;
                -0.244138101655,-0.000001134409,0.000000000000,0.000000000000,0.203823021502,0.427000081451,0.244334183332,0.524543073655,0.617801800258;
                0.283342026405,0.000001316573,0.000000000000,-0.000000000000,-0.236553112989,-0.495568154122,-0.541711787937,0.388060436089,0.417288344168;
                0.000000000000,0.000000000000,1.000000000000,0.000000000000,0.000000000000,0.000000000000,0.000000000000,0.000000000000,0.000000000000;
                    ],
                eps: 4.9e-8,
                zeps: 4.1e-6,
            },
        ];
        for test in Vec::from(&tests[..]) {
            let (n, _m, _a, mut d, e) = tred3(test.inp);
            let mut z = Dmat::identity(n, n);
            dbg!(&test.label);
            tql2(n, e, &mut d, &mut z);
            check_vec!(Dvec::from(d), test.d, test.eps, test.label);
            check_mat!(&z, &test.z, test.zeps, test.label);
        }
    }

    /// this is really `test_symm_eigen_decomp` since trbak3 is the last step
    /// therein. I even call `symm_eigen_decomp` here now
    #[test]
    fn test_trbak3() {
        #[derive(Clone)]
        struct Test {
            label: &'static str,
            inp: Dmat,
            z: Dmat,
            eps: f64,
        }
        let tests = [
            Test {
                label: "c3hcn",
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
                eps: 4e-10,
            },
            Test {
                label: "c3hf",
                inp: dmatrix![
                        66.259393886896206, 0.0, 0.0;
                        0.0000000,81.314604235348298, 0.0;
                        -0.0035385188367391684,0.0000000,15.055210348452093;
                ],
                z: dmatrix![
                    0.000069106049,0.999999997612,0.000000000000;
                    0.000000000000,0.000000000000,-1.000000000000;
                    0.999999997612,-0.000069106049,0.000000000000;
                ],
                eps: 4e-10,
            },
            Test {
                label: "h2o lxm",
                inp: load_dmat("testfiles/h2o/linalg_fxm", 9, 9),
                z: load_dmat("testfiles/h2o/linalg_z_final", 9, 9),
                eps: 1e-6,
            },
            Test {
                label: "h2o sic lxm",
                inp: load_dmat("testfiles/h2o_sic/step_fxm", 9, 9),
                z: load_dmat("testfiles/h2o_sic/linalg_z_final", 9, 9),
                eps: 1e-6,
            },
        ];
        for test in Vec::from(&tests[..]) {
            let (_, z) = symm_eigen_decomp(test.inp, false);
            check_mat!(&z, &test.z, test.eps, test.label);
        }
    }
}

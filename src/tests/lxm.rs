use approx::assert_abs_diff_eq;

use super::*;

#[derive(Clone)]
struct Test {
    infile: String,
    fort15: String,
    nvib: usize,
    lxm: Dmat,
    harm: Vec<f64>,
}

impl Test {
    fn new(dir: &'static str, nvib: usize, harm: Vec<f64>) -> Self {
        let start = Path::new("testfiles");
        Self {
            infile: String::from(
                start.join(dir).join("spectro.in").to_str().unwrap(),
            ),
            fort15: String::from(
                start.join(dir).join("fort.15").to_str().unwrap(),
            ),
            nvib,
            lxm: load_dmat(
                start.join(dir).join("lxm").to_str().unwrap(),
                nvib,
                nvib,
            ),
            harm,
        }
    }
}

#[test]
fn test_lxm() {
    let tests = [
        Test::new(
            "h2o",
            9,
            vec![
                3943.6903070625431,
                3833.7018985135023,
                1650.9329629443762,
                30.024414884101041,
                29.153600077627246,
                7.3277291693141766,
                0.072620302917321009,
                // these two are imaginary so negate
                -0.17306628504479143,
                -29.613852000401547,
            ],
        ),
        Test::new(
            "h2co",
            12,
            vec![
                3004.5902666751244,
                2932.5963462190625,
                1778.656454731436,
                1534.0975439880112,
                1269.7653819557681,
                1186.9127436725194,
                20.7124117145589,
                10.284632778680406,
                6.9046242128632693,
                0.2112502299238036,
                // these two are imaginary so negate
                -0.26310616726935793,
                -0.34056157109382451,
            ],
        ),
        Test::new(
            "c3h2",
            15,
            vec![
                3281.2437096923395,
                3247.5423614827209,
                1623.3244098628386,
                1307.5960052790911,
                1090.6948808317188,
                992.97815025157308,
                908.49025841765001,
                901.52693616201475,
                785.3785617187965,
                15.378822733191582,
                2.7912779094604101,
                0.74085952506782848,
                // imag
                -0.86464536511179046,
                -4.0573479997005943,
                -7.0872468907754147,
            ],
        ),
        Test::new(
            "c3hf",
            15,
            vec![
                3271.2239104039636,
                1820.6004435804787,
                1393.9217365879335,
                1156.2295924892737,
                944.79905255584515,
                909.29032606084388,
                790.44246966076093,
                466.3783495907353,
                465.65704985266359,
                0.011693484738423165,
                0.0077769817780052831,
                // imag
                -0.0083165512465408781,
                -0.0084532136566841092,
                -0.013714840327971822,
                -0.015363142797476336,
            ],
        ),
        Test::new(
            "c3hcn",
            18,
            vec![
                3271.5881522343207,
                2280.1997504340388,
                1728.3660667655636,
                1277.1652514034622,
                1116.2949919685659,
                947.2134830682661,
                904.21228113857137,
                680.772647791606,
                532.87743879614311,
                530.07466622919128,
                221.13564246722288,
                203.53701262964046,
                0.016090303129064221,
                0.0071223184366794815,
                -0.0037808484133202919,
                -0.0086757003866622523,
                -0.011240676534594452,
                -0.016734163693373296,
            ],
        ),
        Test::new(
            "c3hcn010",
            18,
            vec![
                3271.1795755233238,
                2280.0924448154542,
                1728.134307968186,
                1277.1806063711706,
                1116.4518271280579,
                947.10751012130777,
                904.09844247318347,
                680.79594896651645,
                532.75583438253011,
                530.11422631349251,
                221.14178982563286,
                203.562478678129,
                0.012154875257174607,
                0.010985078565658966,
                -0.0090818588711545824,
                -0.0098588933562224201,
                -0.013180931940242106,
                -0.020806633490717621,
            ],
        ),
    ];
    for test in Vec::from(&tests[..]) {
        let s = Spectro::load(&test.infile);
        let fc2 = load_fc2(&test.fort15, test.nvib);
        let fc2 = s.rot2nd(fc2);
        let fc2 = FACT2 * fc2;
        let w = s.geom.weights();
        let sqm: Vec<_> = w.iter().map(|w| 1.0 / w.sqrt()).collect();
        let fxm = s.form_sec(fc2, &sqm);

        let (harms, lxm) = symm_eigen_decomp(fxm);

        assert_abs_diff_eq!(
            to_wavenumbers(harms),
            Dvec::from(test.harm),
            epsilon = 6e-6
        );

        // only really care about the part with frequencies. there is more noise
        // in the rotations and translations, so this allows tightening epsilon
        let got = lxm.slice((0, 0), (s.n3n, s.nvib)).abs();
        let want = test.lxm.slice((0, 0), (s.n3n, s.nvib)).abs();

        // println!("{:.2e}", (got.clone() - want.clone()).max());
        assert_abs_diff_eq!(got, want, epsilon = 2e-9);
    }
}

#[test]
fn test_lx() {
    // just test a hard one here for now
    let s = Spectro::load("testfiles/c3hf/spectro.in");
    let fc2 = load_fc2("testfiles/c3hf/fort.15", 15);
    let fc2 = s.rot2nd(fc2);
    let fc2 = FACT2 * fc2;
    let w = s.geom.weights();
    let sqm: Vec<_> = w.iter().map(|w| 1.0 / w.sqrt()).collect();
    let fxm = s.form_sec(fc2, &sqm);

    let (_harms, lxm) = symm_eigen_decomp(fxm);
    let lx = s.make_lx(s.n3n, &sqm, &lxm);

    let got = lx.slice((0, 0), (s.nvib, s.nvib)).abs();
    let want = load_dmat("testfiles/c3hf/lx", 15, 15);
    let want = want.slice((0, 0), (s.nvib, s.nvib)).abs();

    // println!("{:.2e}", (got.clone() - want.clone()).max());
    assert_abs_diff_eq!(got, want, epsilon = 5e-9);
}

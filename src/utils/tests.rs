use crate::*;

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
        24.0, 6.0, 4.0, 6.0, 24.0, 6.0, 2.0, 2.0, 6.0, 4.0, 2.0, 4.0, 6.0, 6.0,
        24.0,
    ];
    assert_eq!(got, want);
}

//! Phase 102 integration tests: stdlib ML dataset and dataframe.

use iris::{compile_multi, EmitKind};

// ── 1. dataset_mean([1.0, 2.0, 3.0]) → 2.0 ──────────────────────────────────
#[test]
fn test_dataset_mean() {
    let src = r#"
bring std.dataset
def f() -> f64 {
    var data = list()
    push(data, 1.0);
    push(data, 2.0);
    push(data, 3.0);
    dataset_mean(data)
}
"#;
    let result = compile_multi(&[("main", src)], "main", EmitKind::Eval).unwrap();
    let v: f64 = result.trim().parse().unwrap();
    assert!((v - 2.0).abs() < 1e-9, "expected 2.0, got {}", v);
}

// ── 2. dataset_std([1.0, 2.0, 3.0]) correct value ───────────────────────────
#[test]
fn test_dataset_std() {
    let src = r#"
bring std.dataset
def f() -> f64 {
    var data = list()
    push(data, 1.0);
    push(data, 2.0);
    push(data, 3.0);
    dataset_std(data)
}
"#;
    let result = compile_multi(&[("main", src)], "main", EmitKind::Eval).unwrap();
    let v: f64 = result.trim().parse().unwrap();
    // std of [1,2,3] = sqrt(2/3) ≈ 0.8165
    assert!((v - 0.8164965809277261).abs() < 1e-6, "expected ~0.8165, got {}", v);
}

// ── 3. dataset_normalize → zero mean, unit std ──────────────────────────────
#[test]
fn test_dataset_normalize() {
    let src = r#"
bring std.dataset
def f() -> f64 {
    var data = list()
    push(data, 2.0);
    push(data, 4.0);
    push(data, 6.0);
    val norm = dataset_normalize(data)
    dataset_mean(norm)
}
"#;
    let result = compile_multi(&[("main", src)], "main", EmitKind::Eval).unwrap();
    let v: f64 = result.trim().parse().unwrap();
    assert!(v.abs() < 1e-9, "normalized mean should be ~0, got {}", v);
}

// ── 4. dataset_split with 0.6 gives correct sizes ───────────────────────────
#[test]
fn test_dataset_split() {
    let src = r#"
bring std.dataset
def f() -> i64 {
    var data = list()
    push(data, 1.0);
    push(data, 2.0);
    push(data, 3.0);
    push(data, 4.0);
    push(data, 5.0);
    val (train, test) = dataset_split(data, 0.6 to f64)
    list_len(train)
}
"#;
    let result = compile_multi(&[("main", src)], "main", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "3");
}

// ── 5. dataset_batch_count gives correct batch count ────────────────────────
#[test]
fn test_dataset_batch_count() {
    let src = r#"
bring std.dataset
def f() -> i64 {
    var data = list()
    push(data, 1.0);
    push(data, 2.0);
    push(data, 3.0);
    push(data, 4.0);
    push(data, 5.0);
    push(data, 6.0);
    dataset_batch_count(data, 2)
}
"#;
    let result = compile_multi(&[("main", src)], "main", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "3");
}

// ── 6. df_new + df_add_row + df_len correct count ───────────────────────────
#[test]
fn test_df_add_row_len() {
    let src = r#"
bring std.dataframe
def f() -> i64 {
    var cols = list()
    push(cols, "x");
    push(cols, "y");
    val df0 = df_new(cols)
    var r1 = list()
    push(r1, 1.0);
    push(r1, 2.0);
    val df1 = df_add_row(df0, r1)
    var r2 = list()
    push(r2, 3.0);
    push(r2, 4.0);
    val df2 = df_add_row(df1, r2)
    df_len(df2)
}
"#;
    let result = compile_multi(&[("main", src)], "main", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "2");
}

// ── 7. df_col extracts correct column ───────────────────────────────────────
#[test]
fn test_df_col() {
    let src = r#"
bring std.dataframe
def f() -> f64 {
    var cols = list()
    push(cols, "x");
    push(cols, "y");
    val df0 = df_new(cols)
    var r1 = list()
    push(r1, 10.0);
    push(r1, 20.0);
    val df1 = df_add_row(df0, r1)
    var r2 = list()
    push(r2, 30.0);
    push(r2, 40.0);
    val df2 = df_add_row(df1, r2)
    val ys = df_col(df2, "y")
    list_get(ys, 1)
}
"#;
    let result = compile_multi(&[("main", src)], "main", EmitKind::Eval).unwrap();
    let v: f64 = result.trim().parse().unwrap();
    assert!((v - 40.0).abs() < 1e-9, "expected 40.0, got {}", v);
}

// ── 8. df_filter returns subset of rows in range ────────────────────────────
#[test]
fn test_df_filter() {
    let src = r#"
bring std.dataframe
def f() -> i64 {
    var cols = list()
    push(cols, "score");
    val df0 = df_new(cols)
    var r1 = list()
    push(r1, 50.0 to f64);
    val df1 = df_add_row(df0, r1)
    var r2 = list()
    push(r2, 80.0 to f64);
    val df2 = df_add_row(df1, r2)
    var r3 = list()
    push(r3, 95.0 to f64);
    val df3 = df_add_row(df2, r3)
    val high = df_filter(df3, "score", 75.0 to f64, 100.0 to f64)
    df_len(high)
}
"#;
    let result = compile_multi(&[("main", src)], "main", EmitKind::Eval).unwrap();
    assert_eq!(result.trim(), "2");
}

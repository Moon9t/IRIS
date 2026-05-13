use iris::stdlib::stdlib_source;

#[test]
fn stdlib_sql_module_exports_param_helpers() {
    let src = stdlib_source("sql").expect("sql stdlib source");
    assert!(src.contains("sql_query_params"));
    assert!(src.contains("sql_exec_params"));
    assert!(src.contains("sql_query_xy_params"));
}

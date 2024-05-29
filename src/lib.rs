use pyo3::prelude::*;
//use std::collections::HashMap;
use serde::Deserialize;
// use serde_json;
use pyo3::types::{PyDict, PyList};
use polars::prelude::*;
use pyo3_polars::PyDataFrame;

mod time_domain_processing;

/// A Python module implemented in Rust.
#[pymodule]
fn rcs_cataloging(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(time_domain_processing::loop_and_table_td_data, m)?)?;
    Ok(())
}



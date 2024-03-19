use pyo3::prelude::*;
//use std::collections::HashMap;
use serde::Deserialize;
// use serde_json;
use pyo3::types::{PyDict, PyList};
use polars::prelude::*;
use pyo3_polars::PyDataFrame;

#[derive(Deserialize)]
struct TimeDomainPacket {
    Header: Header,
    PacketGenTime: i64,
    PacketRxUnixTime: i64,
    ChannelSamples: Vec<ChannelSample>,
    DebugInfo: i64,
    EvokedMarker: Vec<i64>,
    IncludedChannels: i64,
    SampleRate: i64,
    Units: String,
}

#[derive(Deserialize)]
struct Header {
    dataSize: i64,
    dataType: i64,
    dataTypeSequence: i64,
    globalSequence: i64,
    info: i64,
    systemTick: i64,
    timestamp: Timestamp,
    user1: i64,
    user2: i64,
}

#[derive(Deserialize)]
struct Timestamp {
    seconds: i64,
}

#[derive(Deserialize)]
struct ChannelSample {
    Key: i64, // Consider Enum for Key so that it can be Key::Channel0, Key::Channel1, etc...
    Value: Vec<f64>,
}

// Implement the `FromPyObject` trait for `TimeDomainPacket`
impl<'a> FromPyObject<'a> for TimeDomainPacket {
    fn extract(obj: &'a PyAny) -> PyResult<Self> {
        let dict = obj.downcast::<PyDict>()?;

        let header_dict = dict.get_item("Header").unwrap().expect("REASON").downcast::<PyDict>()?;
        let header = Header {
            dataSize: header_dict.get_item("dataSize").unwrap().expect("REASON").extract()?,
            dataType: header_dict.get_item("dataType").unwrap().expect("REASON").extract()?,
            dataTypeSequence: header_dict.get_item("dataTypeSequence").unwrap().expect("REASON").extract()?,
            globalSequence: header_dict.get_item("globalSequence").unwrap().expect("REASON").extract()?,
            info: header_dict.get_item("info").unwrap().expect("REASON").extract()?,
            systemTick: header_dict.get_item("systemTick").unwrap().expect("REASON").extract()?,
            timestamp: Timestamp {
                seconds: header_dict.get_item("timestamp").unwrap().expect("REASON").downcast::<PyDict>()?.get_item("seconds").unwrap().expect("REASON").extract()?,
            },
            user1: header_dict.get_item("user1").unwrap().expect("REASON").extract()?,
            user2: header_dict.get_item("user2").unwrap().expect("REASON").extract()?,
        };

        let packet_gen_time = dict.get_item("PacketGenTime").unwrap().expect("REASON").extract()?;
        let packet_rx_unix_time = dict.get_item("PacketRxUnixTime").unwrap().expect("REASON").extract()?;

        let channel_samples = dict.get_item("ChannelSamples").unwrap().expect("REASON").downcast::<PyList>()?.iter().map(|obj| {
            let sample_dict = obj.downcast::<PyDict>().unwrap();
            ChannelSample {
                Key: sample_dict.get_item("Key").unwrap().expect("REASON").extract().expect("REASON"),
                Value: sample_dict.get_item("Value").unwrap().expect("REASON").downcast::<PyList>().unwrap().extract().unwrap(),
            }
        }).collect::<Vec<_>>();

        let debug_info = dict.get_item("DebugInfo").unwrap().expect("REASON").extract()?;
        let evoked_marker = dict.get_item("EvokedMarker").unwrap().expect("REASON").downcast::<PyList>()?.extract::<Vec<i64>>()?;
        let included_channels = dict.get_item("IncludedChannels").unwrap().expect("REASON").extract()?;
        let sample_rate = dict.get_item("SampleRate").unwrap().expect("REASON").extract()?;
        let units = dict.get_item("Units").unwrap().expect("REASON").extract()?;

        Ok(TimeDomainPacket {
            Header: header,
            PacketGenTime: packet_gen_time,
            PacketRxUnixTime: packet_rx_unix_time,
            ChannelSamples: channel_samples,
            DebugInfo: debug_info,
            EvokedMarker: evoked_marker,
            IncludedChannels: included_channels,
            SampleRate: sample_rate,
            Units: units,
        })
    }
}

#[pyfunction]
fn loop_and_table_td_data(py: Python, data_list: Vec<PyObject>) -> PyResult<PyObject> {
    let mut data = Vec::<TimeDomainPacket>::new();
    for item in data_list {
        let time_domain_packet: TimeDomainPacket = item.extract::<TimeDomainPacket>(py).expect("REASON");
        data.push(time_domain_packet);
    }

    let df = DataFrame::new(vec![
        Series::new("timestamp", data.iter().map(|p| p.Header.timestamp.seconds).collect::<Vec<_>>()),
        Series::new("packet_gen_time", data.iter().map(|p| p.PacketGenTime).collect::<Vec<_>>()),
        Series::new("packet_rx_unix_time", data.iter().map(|p| p.PacketRxUnixTime).collect::<Vec<_>>()),
        //Series::new("channel_samples", data.iter().map(|p| p.ChannelSamples).collect::<Vec<_>>()),
        Series::new("SampleRate", data.iter().map(|p| p.DebugInfo).collect::<Vec<_>>()),
        // Add more columns as needed for the remaining fields
    ]);

    // Convert the Result<DataFrame, PolarsError> to a PyObject
    let result = match df {
        Ok(df) => {
            let pydf = PyDataFrame(df);
            let py_obj = pydf.into_py(py);
            Ok(py_obj)
        }
        Err(err) => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Polars error: {}", err))),
    };

    result
}

/// A Python module implemented in Rust.
#[pymodule]
fn rcs_cataloging(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(loop_and_table_td_data, m)?)?;
    Ok(())
}



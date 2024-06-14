use pyo3::prelude::*;
//use std::collections::HashMap;
use serde::Deserialize;
// use serde_json;
use pyo3::types::{PyDict, PyList};
use polars::prelude::*;
use pyo3_polars::PyDataFrame;
use chrono::{TimeZone, Utc, Local};

#[derive(Deserialize)]
struct AccelPacket {
    Header: Header,
    PacketGenTime: i64,
    PacketRxUnixTime: i64,
    XSamples: Vec<f64>,
    YSamples: Vec<f64>,
    ZSamples: Vec<f64>,
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

// Implement the `FromPyObject` trait for `TimeDomainPacket`
impl<'a> FromPyObject<'a> for AccelPacket {
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

        let x_samples = dict.get_item("XSamples").unwrap().expect("REASON").downcast::<PyList>()?.extract::<Vec<f64>>()?;
        let y_samples = dict.get_item("YSamples").unwrap().expect("REASON").downcast::<PyList>()?.extract::<Vec<f64>>()?;
        let z_samples = dict.get_item("ZSamples").unwrap().expect("REASON").downcast::<PyList>()?.extract::<Vec<f64>>()?;

        // let channel_samples = dict.get_item("ChannelSamples").unwrap().expect("REASON").downcast::<PyList>()?.iter().map(|obj| {
        //     let sample_dict = obj.downcast::<PyDict>().unwrap();
        //     ChannelSample {
        //         Key: sample_dict.get_item("Key").unwrap().expect("REASON").extract().expect("REASON"),
        //         Value: sample_dict.get_item("Value").unwrap().expect("REASON").downcast::<PyList>().unwrap().extract().unwrap(),
        //     }
        // }).collect::<Vec<_>>();

        let sample_rate = dict.get_item("SampleRate").unwrap().expect("REASON").extract()?;
        let units = dict.get_item("Units").unwrap().expect("REASON").extract()?;

        Ok(AccelPacket {
            Header: header,
            PacketGenTime: packet_gen_time,
            PacketRxUnixTime: packet_rx_unix_time,
            XSamples: x_samples,
            YSamples: y_samples,
            ZSamples: z_samples,
            SampleRate: sample_rate,
            Units: units,
        })
    }
}

#[pyfunction]
pub fn loop_and_table_accel_data(py: Python, data_list: Vec<PyObject>) -> PyResult<PyObject> {
    let mut data = Vec::<AccelPacket>::new();
    for item in data_list {
        let time_domain_packet: AccelPacket = item.extract::<AccelPacket>(py).expect("REASON");
        data.push(time_domain_packet);
    }

    // let packet_size = data[0].XSamples.len();

    // let XSamples = data.iter().map(|p| Series::new("", &p.XSamples)).collect::<Vec<_>>();
    // let YSamples = data.iter().map(|p| Series::new("", &p.YSamples)).collect::<Vec<_>>();
    // let ZSamples = data.iter().map(|p| Series::new("", &p.ZSamples)).collect::<Vec<_>>();


    let df = DataFrame::new(vec![
        // Series::new("localTime", data.iter().map(|p| Utc.timestamp(p.PacketRxUnixTime / 1000, (p.PacketRxUnixTime % 1000 * 1_000_000).try_into().unwrap()).with_timezone(&Local).format("%Y-%m-%d %H:%M:%S%.3f").to_string()).collect::<Vec<_>>()),
        Series::new("timestamp", data.iter().map(|p| p.Header.timestamp.seconds).collect::<Vec<_>>()),
        Series::new("PacketGenTime", data.iter().map(|p| p.PacketGenTime).collect::<Vec<_>>()),
        Series::new("PacketRxUnixTime", data.iter().map(|p| p.PacketRxUnixTime).collect::<Vec<_>>()),
        Series::new("systemTick", data.iter().map(|p| p.Header.systemTick).collect::<Vec<_>>()),
        Series::new("dataTypeSequence", data.iter().map(|p| p.Header.dataTypeSequence).collect::<Vec<_>>()),
        Series::new("samplerate", data.iter().map(|p| p.SampleRate).collect::<Vec<_>>()),
        Series::new("XSamples", data.iter().map(|p| Series::new("", &p.XSamples)).collect::<Vec<_>>()),
        Series::new("YSamples", data.iter().map(|p| Series::new("", &p.YSamples)).collect::<Vec<_>>()),
        Series::new("ZSamples", data.iter().map(|p| Series::new("", &p.ZSamples)).collect::<Vec<_>>()),
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
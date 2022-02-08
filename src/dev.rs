use std::fs::{self, File};
use std::io::Read;
#[macro_use]
extern crate log;
use env_logger::{Builder, Env};
use tract_onnx::prelude::tract_ndarray;

use c2c7demo::{extract_points, PointData, LABELS};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let env = Env::default().filter_or("LOG_LEVEL", "debug");
    Builder::from_env(env)
        .format_timestamp(Some(env_logger::TimestampPrecision::Seconds))
        .init();
    let filename = "result.bin";
    let (chs, height, width) = (6, 768, 768);
    let mut f = File::open(&filename).expect("no file found");
    let metadata = fs::metadata(&filename).expect("unable to read metadata");
    let mut buffer = vec![0; metadata.len() as usize];
    f.read(&mut buffer).expect("buffer overflow");
    if buffer.len() != chs * height * width {
        panic!("Invalid input size")
    } else {
        debug!("{} loaded.", filename);
    }
    let arr = tract_ndarray::Array3::from_shape_vec((chs, height, width), buffer)?;
    debug!("array shape: {:?}", arr.shape());
    let optimal_points = extract_points(&arr);
    debug!("Optimal points\n{:?}", optimal_points);
    let labels: Vec<String> = LABELS.iter().map(|s| String::from(*s)).collect();
    let data = PointData::new(&optimal_points, &labels, width, height, "");
    data.save("tmp.json")?;
    return Ok(());
}

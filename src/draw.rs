use serde::{Deserialize, Serialize};
// use serde_json::Result;
use svg::node::element::path::Data;
use svg::node::element::Path;
use svg::Document;

type Point = (f32, f32);

#[derive(Serialize, Deserialize, Debug)]
struct Flags {}

#[derive(Serialize, Deserialize, Debug)]
struct Shape {
    label: String,
    points: Vec<Point>,
    group_id: Option<String>,
    shape_type: String,
    flags: Flags,
}

#[derive(Serialize, Deserialize, Debug)]
struct PointData {
    version: String,
    flags: Flags,
    shapes: Vec<Shape>,
    imagePath: String,
    imageData: Option<String>,
    imageHeight: usize,
    imageWidth: usize,
}

use std::fs::File;
use std::io::prelude::*;
use std::io::BufReader;


fn main() -> Result<(), Box<dyn std::error::Error>> {
    let filename = "dicom.json";
    let file = File::open(filename)?;
    let reader = BufReader::new(file);
    let json_data: PointData = serde_json::from_reader(reader)
        .or_else(|err| Err(Box::new(err) as Box<dyn std::error::Error>))?;
    println!("{:?}", json_data);
    // let data = Data::new()
    //     .move_to((10, 10))
    //     .line_by((0, 50))
    //     .line_by((50, 0))
    //     .line_by((0, -50))
    //     .close();

    // let path = Path::new()
    //     .set("fill", "none")
    //     .set("stroke", "black")
    //     .set("stroke-width", 3)
    //     .set("d", data);

    // let document = Document::new().set("viewBox", (0, 0, 70, 70)).add(path);

    // svg::save("image.svg", &document).unwrap();
    return Ok(());
}

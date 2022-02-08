use serde::{Deserialize, Serialize};
// use serde_json::Result;
use std::collections::HashMap;
use svg::node::element::path::Data;
use svg::node::element::{self, Circle, Text};
use svg::node::{self, Value};
use svg::Document;
#[macro_use]
extern crate log;
use env_logger::{Builder, Env};

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
    let env = Env::default().filter_or("LOG_LEVEL", "debug");
    Builder::from_env(env)
        .format_timestamp(Some(env_logger::TimestampPrecision::Seconds))
        .init();

    let filename = "dicom.json";
    let file = File::open(filename)?;
    let reader = BufReader::new(file);
    let json_data: PointData = serde_json::from_reader(reader)
        .or_else(|err| Err(Box::new(err) as Box<dyn std::error::Error>))?;
    println!("{:?}", json_data);
    let (image_width, image_height) = (json_data.imageWidth, json_data.imageHeight);
    let shapes = json_data.shapes;
    assert!(shapes.len() >= 4);
    let mut shape_map = HashMap::new();
    for shape in shapes.into_iter() {
        println!("{:?}", shape);
        assert_eq!(shape.points.len(), 1);
        shape_map.insert(shape.label, shape.points[0]);
    }

    let labels = vec!["C2A", "C2P", "C7A", "C7P"];
    let colors = vec!["red", "lime", "blue", "magenta"];
    let mut document = Document::new()
        .set("width", image_width)
        .set("height", image_height)
        .set("viewBox", (0, 0, image_width, image_height));

    let mut buf = Vec::new();
    let base_img = image::open("dicom.png").unwrap();
    base_img
        .write_to(&mut buf, image::ImageOutputFormat::Png)
        .unwrap();
    let res_base64 = base64::encode(&buf);
    let b64 = "data:image/png;base64,".to_owned() + &res_base64;
    let bg = element::Image::new()
        .set("x", 0)
        .set("y", 0)
        .set("width", image_width)
        .set("height", image_height)
        .set("href", b64);
    document = document.add(bg);

    let point_radius = "10";
    let point_color = "red";
    let font_size = "64";
    let font_family = "serif";
    for (label, color) in labels.into_iter().zip(colors) {
        let point_xy = shape_map[label];
        let circle = Circle::new()
            .set("fill", color)
            .set("stroke", "black")
            .set("stroke-width", 0)
            .set("cx", point_xy.0)
            .set("cy", point_xy.1)
            .set("r", point_radius);
        let text_node = node::Text::new(label);
        let text = element::Text::new()
            .set("x", point_xy.0)
            .set("y", point_xy.1)
            .set("font-family", font_family)
            .set("font-size", font_size)
            .set("fill", "black")
            .add(text_node);
        document = document.add(circle).add(text);
    }
    svg::save("wbg.svg", &document).unwrap();
    // let rtree = resvg::usvg::
    // resvg::render()
    // use resvg::usvg;
    let mut opt = usvg::Options::default();
    // Get file's absolute directory.
    // opt.resources_dir = std::fs::canonicalize(&args[1]).ok().and_then(|p| p.parent().map(|p| p.to_path_buf()));
    opt.fontdb.load_system_fonts();
    let rtree = usvg::Tree::from_str(&document.to_string(), &opt.to_ref()).unwrap();
    let pixmap_size = rtree.svg_node().size.to_screen_size();
    let mut pixmap = tiny_skia::Pixmap::new(pixmap_size.width(), pixmap_size.height()).unwrap();
    resvg::render(
        &rtree,
        usvg::FitTo::Original,
        tiny_skia::Transform::default(),
        pixmap.as_mut(),
    )
    .unwrap();
    pixmap.save_png("rendered.png").unwrap();

    // let mut buf = Vec::new();
    // svg::write(buf, &document).unwrap();
    // let svg_string = String::from_utf8(buf).unwrap();

    // svg::save("image.svg", &document).unwrap();
    // svg::
    return Ok(());
}

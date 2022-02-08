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

fn img2base64(img: &image::GrayImage) -> String {
    let mut buf = Vec::new();
    let base_img = image::open("dicom.png").unwrap();
    base_img
        .write_to(&mut buf, image::ImageOutputFormat::Png)
        .unwrap();
    base64::encode(&buf)
}

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
    let (image_width, image_height) = (json_data.imageWidth, json_data.imageHeight);
    let shapes = json_data.shapes;
    assert!(shapes.len() >= 4);
    let mut shape_map = HashMap::new();
    for shape in shapes.into_iter() {
        assert_eq!(shape.points.len(), 1);
        debug!("{}: {:?}", shape.label, shape.points[0]);
        shape_map.insert(shape.label, shape.points[0]);
    }

    let mut document = Document::new()
        .set("width", image_width)
        .set("height", image_height)
        .set("viewBox", (0, 0, image_width, image_height));

    // set background
    let base_img = image::open("dicom.png").unwrap();
    let res_base64 = img2base64(&base_img.to_luma8());
    let b64 = "data:image/png;base64,".to_owned() + &res_base64;
    let bg = element::Image::new()
        .set("x", 0)
        .set("y", 0)
        .set("width", image_width)
        .set("height", image_height)
        .set("href", b64);
    document = document.add(bg);

    let labels = vec!["C2A", "C2P", "C7A", "C7P"];
    // draw lines
    let line_width = 10;
    let line_color = "yellow";
    let image_box = lyon_geom::Box2D::new(
        lyon_geom::Point::new(0f32, 0f32),
        lyon_geom::Point::new(image_width as f32, image_height as f32),
    );
    let tl = lyon_geom::Point::new(0f32, 0f32);
    let tr = lyon_geom::Point::new(image_width as f32, 0f32);
    let top_line = lyon_geom::Line {
        point: tl,
        vector: tr - tl,
    };

    let bl = lyon_geom::Point::new(0f32, image_height as f32);
    let br = lyon_geom::Point::new(image_width as f32, image_height as f32);
    let bottom_line = lyon_geom::Line {
        point: bl,
        vector: br - bl,
    };
    let left_line = lyon_geom::Line {
        point: tl,
        vector: bl - tl,
    };
    let right_line = lyon_geom::Line {
        point: tr,
        vector: br - tr,
    };
    let points: Vec<Point> = labels.iter().map(|l| shape_map[*l]).collect();
    let mut group = element::Group::new()
        .set("stroke", line_color)
        .set("stroke-width", line_width);
    for (p1, p2) in vec![(points[0], points[1]), (points[2], points[3])] {
        let p = lyon_geom::Point::new(p1.0, p1.1);
        let v = p - lyon_geom::Point::new(p2.0, p2.1);
        let line = lyon_geom::Line {
            point: p,
            vector: v,
        };
        let (int1, int2) = if (v.y / v.x) > 0.5 {
            (
                line.intersection(&top_line).unwrap(),
                line.intersection(&bottom_line).unwrap(),
            )
        } else {
            (
                line.intersection(&left_line).unwrap(),
                line.intersection(&right_line).unwrap(),
            )
        };
        let line = element::Line::new()
            .set("x1", int1.x)
            .set("y1", int1.y)
            .set("x2", int2.x)
            .set("y2", int2.y);
        group = group.add(line);
    }
    document = document.add(group);

    // draw points
    let colors = vec!["red", "lime", "blue", "magenta"];
    let point_radius = "10";
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
    let mut opt = usvg::Options::default();
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
    debug!("Render");
    pixmap.save_png("rendered.png").unwrap();
    return Ok(());
}

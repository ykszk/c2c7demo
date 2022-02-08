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

trait FromPoint<S> {
    fn from_points(p1: lyon_geom::Point<S>, p2: lyon_geom::Point<S>) -> Self
    where
        S: std::ops::Sub<Output = S> + Clone;
}

impl<S> FromPoint<S> for lyon_geom::Line<S> {
    fn from_points(p1: lyon_geom::Point<S>, p2: lyon_geom::Point<S>) -> Self
    where
        S: std::ops::Sub<Output = S> + Clone,
    {
        let v = p2 - p1.clone();
        lyon_geom::Line {
            point: p1,
            vector: v,
        }
    }
}

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
#[allow(non_snake_case)]
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

fn img2base64(img: &image::DynamicImage) -> String {
    let mut buf = Vec::new();
    img.write_to(&mut buf, image::ImageOutputFormat::Png)
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
    let res_base64 = img2base64(&base_img);
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
    let tl = lyon_geom::Point::new(0f32, 0f32);
    let tr = lyon_geom::Point::new(image_width as f32, 0f32);
    let bl = lyon_geom::Point::new(0f32, image_height as f32);
    let br = lyon_geom::Point::new(image_width as f32, image_height as f32);
    let top_line = lyon_geom::Line::from_points(tl, tr);
    let bottom_line = lyon_geom::Line::from_points(bl, br);
    let left_line = lyon_geom::Line::from_points(tl, bl);
    let right_line = lyon_geom::Line::from_points(tr, br);
    let points: Vec<Point> = labels.iter().map(|l| shape_map[*l]).collect();
    let mut group = element::Group::new()
        .set("stroke", line_color)
        .set("stroke-width", line_width);

    let (p1, p2) = (points[0], points[1]);
    let c2_line = lyon_geom::Line::from_points(
        lyon_geom::Point::new(p1.0, p1.1),
        lyon_geom::Point::new(p2.0, p2.1),
    );
    let (p1, p2) = (points[2], points[3]);
    let c7_line = lyon_geom::Line::from_points(
        lyon_geom::Point::new(p1.0, p1.1),
        lyon_geom::Point::new(p2.0, p2.1),
    );
    for line in vec![c2_line, c7_line] {
        let v = &line.vector;
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
    let angle_degree = c2_line.vector.angle_to(c7_line.vector).to_degrees();
    let intersect = c2_line.intersection(&c7_line).unwrap();
    let sign = if intersect.x > points[0].0 { 1.0 } else { -1.0 };
    let angle_degree = sign * angle_degree;
    let angle = format!("{:.1}°", angle_degree);
    debug!("angle {}", angle);

    let font_size = "64";
    let font_family = "serif";

    if lyon_geom::Box2D::new(tl, br).contains(intersect) {
        debug!("Inside");
        let text_node = node::Text::new(angle);
        let text = element::Text::new()
            .set("x", intersect.x)
            .set("y", intersect.y)
            .set("font-family", font_family)
            .set("font-size", font_size)
            .set("fill", "black")
            .add(text_node);
        document = document.add(text);
    } else {
        debug!("Draw aux lines");
        let a = lyon_geom::Point::new(points[0].0, points[0].1);
        let p = lyon_geom::Point::new(points[1].0, points[1].1);
        let aux_c2 = p + (p - a) / 2.0;
        let t_origin = lyon_geom::Transform::translation(-intersect.x, -intersect.y);
        let mut rog_angle = c2_line.vector.angle_to(c7_line.vector);
        rog_angle.radians /= 2.0;
        let t = t_origin
            .then_rotate(rog_angle)
            .then_translate(lyon_geom::Vector::new(intersect.x, intersect.y));
        let aux_int = t.transform_point(aux_c2);
        debug!("Aux intersection {:?}", aux_int);
        for (line_self, line_other) in vec![(c2_line, c7_line), (c7_line, c2_line)] {
            let aux_on_line = line_self.equation().project_point(&aux_int);
            let aux_p1 = aux_on_line.lerp(
                lyon_geom::Line::from_points(aux_on_line, aux_int)
                    .intersection(&line_other)
                    .unwrap(),
                0.75,
            );
            let line = element::Line::new()
                .set("x1", aux_p1.x)
                .set("y1", aux_p1.y)
                .set("x2", aux_on_line.x)
                .set("y2", aux_on_line.y);
            group = group.add(line);
        }
        let text_node = node::Text::new(angle);
        let text = element::Text::new()
            .set("x", aux_int.x)
            .set("y", aux_int.y)
            .set("font-family", font_family)
            .set("font-size", font_size)
            .set("fill", "black")
            .add(text_node);
        document = document.add(text);
    }

    document = document.add(group);

    // draw points
    let colors = vec!["red", "lime", "blue", "magenta"];
    let point_radius = "10";
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

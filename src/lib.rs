use std::collections::HashMap;

use serde::{Deserialize, Serialize};
#[macro_use]
extern crate log;
use svg::node;
use svg::node::element::{self, Circle};

pub type Point = (f32, f32);
pub const LABELS: [&str; 4] = ["C2A", "C2P", "C7A", "C7P"];

pub trait FromPoint<S> {
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
pub struct Flags {}

#[derive(Serialize, Deserialize, Debug)]
pub struct Shape {
    pub label: String,
    pub points: Vec<Point>,
    pub group_id: Option<String>,
    pub shape_type: String,
    pub flags: Flags,
}

#[derive(Serialize, Deserialize, Debug)]
#[allow(non_snake_case)]
pub struct PointData {
    pub version: String,
    pub flags: Flags,
    pub shapes: Vec<Shape>,
    pub imagePath: String,
    pub imageData: Option<String>,
    pub imageHeight: usize,
    pub imageWidth: usize,
}

impl PointData {
    pub fn new(
        points: &[Point],
        labels: &[String],
        width: usize,
        height: usize,
        path: &str,
    ) -> Self {
        let shapes: Vec<Shape> = points
            .iter()
            .zip(labels)
            .map(|(p, l)| Shape {
                label: l.into(),
                points: vec![*p],
                group_id: None,
                shape_type: "point".into(),
                flags: Flags {},
            })
            .collect();
        Self {
            version: "4.5.7".into(),
            flags: Flags {},
            shapes,
            imagePath: path.into(),
            imageData: None,
            imageHeight: height,
            imageWidth: width,
        }
    }

    pub fn save(&self, filename: &str) -> Result<(), Box<dyn std::error::Error>> {
        let writer = std::io::BufWriter::new(std::fs::File::create(filename)?);
        serde_json::to_writer_pretty(writer, self)
            .map_err(|err| Box::new(err) as Box<dyn std::error::Error>)
    }
}

fn img2base64(img: &image::DynamicImage) -> String {
    let mut buf = Vec::new();
    img.write_to(&mut buf, image::ImageOutputFormat::Png)
        .unwrap();
    base64::encode(&buf)
}

pub fn draw(json_data: PointData, background: Option<String>) -> svg::Document {
    let (image_width, image_height) = (json_data.imageWidth, json_data.imageHeight);
    let shapes = json_data.shapes;
    assert!(shapes.len() >= 4);
    let mut shape_map = HashMap::new();
    for shape in shapes.into_iter() {
        assert_eq!(shape.points.len(), 1);
        debug!("{}: {:?}", shape.label, shape.points[0]);
        shape_map.insert(shape.label, shape.points[0]);
    }

    let mut document = svg::Document::new()
        .set("width", image_width)
        .set("height", image_height)
        .set("viewBox", (0i64, 0i64, image_width, image_height));

    // set background
    if let Some(bg_filename) = background {
        let base_img = image::open(bg_filename).unwrap();
        let res_base64 = img2base64(&base_img);
        let b64 = "data:image/png;base64,".to_owned() + &res_base64;
        let bg = element::Image::new()
            .set("x", 0i64)
            .set("y", 0i64)
            .set("width", image_width)
            .set("height", image_height)
            .set("href", b64);
        document = document.add(bg);
    }

    // draw lines
    let line_width = 10i32;
    let line_color = "yellow";
    let tl = lyon_geom::Point::new(0f32, 0f32);
    let tr = lyon_geom::Point::new(image_width as f32, 0f32);
    let bl = lyon_geom::Point::new(0f32, image_height as f32);
    let br = lyon_geom::Point::new(image_width as f32, image_height as f32);
    let top_line = lyon_geom::Line::from_points(tl, tr);
    let bottom_line = lyon_geom::Line::from_points(bl, br);
    let left_line = lyon_geom::Line::from_points(tl, bl);
    let right_line = lyon_geom::Line::from_points(tr, br);
    let points: Vec<Point> = LABELS.iter().map(|l| shape_map[*l]).collect();
    let mut group = element::Group::new()
        .set("fill", "none")
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
    for line in [c2_line, c7_line] {
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
    let mut angle_text_position = intersect;

    if lyon_geom::Box2D::new(tl, br).contains(intersect) {
        debug!("Inside");
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
        for (line_self, line_other) in [(c2_line, c7_line), (c7_line, c2_line)] {
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
        // draw arcs
        for (i, (line_self, line_other)) in [(c2_line, c7_line), (c7_line, c2_line)]
            .into_iter()
            .enumerate()
        {
            let aux_on_line = line_self.equation().project_point(&aux_int);
            let start = aux_on_line.lerp(
                lyon_geom::Line::from_points(aux_on_line, aux_int)
                    .intersection(&line_other)
                    .unwrap(),
                0.70,
            );
            let radius = (start - aux_int).length();
            let rot_sign = if i == 0 { 1.0 } else { -1.0 };
            let mut rot_angle = c2_line.vector.angle_to(c7_line.vector);
            rot_angle.radians *= rot_sign;
            let t_origin = lyon_geom::Transform::translation(-aux_int.x, -aux_int.y);
            let t = t_origin
                .then_rotate(rot_angle)
                .then_translate(lyon_geom::Vector::new(aux_int.x, aux_int.y));
            let end = t.transform_point(start) - start;
            let mut data = element::path::Data::new().move_to((start.x, start.y));
            let slope = line_self.vector.y.atan2(line_self.vector.x).to_degrees();
            let f1 = if rot_angle.radians.abs() > 180.0 {
                1
            } else {
                0
            } as i32;
            let f2 = if i == 0 { 0 } else { 1 } as i32;
            data = data.elliptical_arc_by((radius, radius, slope, f1, f2, end.x, end.y));
            let path = element::Path::new().set("d", data);
            group = group.add(path);
        }

        angle_text_position = aux_int;
    }

    document = document.add(group);

    let text_node = node::Text::new(angle);
    let text = element::Text::new()
        .set("x", angle_text_position.x)
        .set("y", angle_text_position.y)
        .set("font-family", font_family)
        .set("font-size", font_size)
        .set("fill", "black")
        .add(text_node);
    document = document.add(text);

    // draw points
    let colors = vec!["red", "lime", "blue", "magenta"];
    let point_radius = "10";
    for (label, color) in LABELS.iter().zip(colors) {
        let point_xy = shape_map[*label];
        let circle = Circle::new()
            .set("fill", color)
            .set("stroke", "black")
            .set("stroke-width", 0)
            .set("cx", point_xy.0)
            .set("cy", point_xy.1)
            .set("r", point_radius);
        let text_node = node::Text::new(*label);
        let text = element::Text::new()
            .set("x", point_xy.0)
            .set("y", point_xy.1)
            .set("font-family", font_family)
            .set("font-size", font_size)
            .set("fill", "black")
            .add(text_node);
        document = document.add(circle).add(text);
    }
    document
}

use imageproc::region_labelling::{connected_components, Connectivity};
use tract_onnx::prelude::tract_ndarray::{self, s, Axis};
pub fn extract_points(arr: &tract_ndarray::Array3<u8>) -> Vec<Point> {
    let height = arr.shape()[1];
    let width = arr.shape()[2];
    let ch_axis = Axis(0);
    let mut candidates: Vec<Vec<Point>> = Vec::new();
    for (i_ch, img_ch) in arr.axis_iter(ch_axis).enumerate() {
        if i_ch > 3 {
            // skip affinity maps
            break;
        }
        let max_value = img_ch.iter().max().unwrap();
        let thresh = max_value / 2;
        debug!(
            "Max value for ch {} is {}. thresh is {}",
            i_ch, max_value, thresh
        );
        let bin_arr = img_ch.mapv(|v| if v > thresh { 1u8 } else { 0u8 });
        let bin_img =
            image::GrayImage::from_raw(width as _, height as _, bin_arr.as_slice().unwrap().into())
                .unwrap();
        let cced = connected_components(&bin_img, Connectivity::Four, image::Luma([0u8]));
        let n_cc = *cced.iter().max().unwrap();
        assert!(n_cc > 0);
        debug!("# of CC is {}", n_cc);

        let mut cand_points: Vec<Point> = Vec::new();
        for cc_val in 1..(n_cc + 1) {
            let cc_max_value = img_ch
                .indexed_iter()
                .filter_map(|((y, x), val)| {
                    if cced.get_pixel(x as _, y as _)[0] == cc_val {
                        Some(*val)
                    } else {
                        None
                    }
                })
                .max()
                .unwrap();
            let mut argmax: Vec<(usize, usize)> = Vec::new();
            for ((x, y), val) in img_ch.indexed_iter() {
                if *val == cc_max_value {
                    argmax.push((x, y));
                }
            }
            assert!(
                !argmax.is_empty(),
                "No argmax (coordinates with maximum value) found."
            );
            let sum_argmax: (usize, usize) = argmax
                .iter()
                .fold((0, 0), |acc, x| (acc.0 + x.0, acc.1 + x.1));
            let mean_argmax = (
                sum_argmax.0 as f32 / argmax.len() as f32,
                sum_argmax.1 as f32 / argmax.len() as f32,
            );
            cand_points.push(mean_argmax);
        }
        assert!(!cand_points.is_empty());
        candidates.push(cand_points);
    }
    assert_eq!(candidates.len(), 4);
    debug!("Candidates {:?}", candidates);

    let mut optimal_points: Vec<Point> = Vec::new();
    for c2c7 in 0..2 {
        let cand_left = &candidates[c2c7 * 2];
        let cand_right = &candidates[c2c7 * 2 + 1];
        if cand_left.len() == 1 && cand_right.len() == 1 {
            debug!("No need to find optimal point pairs for {}", c2c7);
            optimal_points.push(cand_left[0]);
            optimal_points.push(cand_right[0]);
        } else {
            debug!("Find optimal point pairs");
            let affinity = arr.slice(s![c2c7 + 4, .., ..]);
            let mut scores: Vec<(usize, Point, Point)> =
                Vec::with_capacity(cand_left.len() * cand_right.len());
            for left in cand_left.iter() {
                for right in cand_right.iter() {
                    let l = (left.0 as isize, left.1 as isize);
                    let r = (right.0 as isize, right.1 as isize);
                    let score: usize = line_drawing::Bresenham::new(l, r)
                        .map(|(x, y)| affinity[[y as usize, x as usize]] as usize)
                        .sum();
                    scores.push((score, *left, *right));
                }
            }
            let (_optimal_score, left, right) = scores.into_iter().max_by_key(|t| t.0).unwrap();
            optimal_points.push(left);
            optimal_points.push(right);
        }
    }
    optimal_points
}

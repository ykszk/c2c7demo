use std::collections::HashMap;
use std::error::Error;
use std::io::Cursor;

use serde::{Deserialize, Serialize};
#[macro_use]
extern crate log;
use svg::node;
use svg::node::element::{self, Circle};

use imageproc::region_labelling::{connected_components, Connectivity};
use tract_onnx::prelude::tract_ndarray as ndarray;
use tract_onnx::prelude::tract_ndarray::{s, Axis};

pub type Point = (f32, f32);
pub const LABELS: [&str; 4] = ["C2A", "C2P", "C7A", "C7P"];
pub const DEFAULT_MODEL: &str = "c2c7.onnx";

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

    pub fn save(&self, filename: &str) -> Result<(), Box<dyn Error>> {
        let writer = std::io::BufWriter::new(std::fs::File::create(filename)?);
        serde_json::to_writer_pretty(writer, self).map_err(|err| Box::new(err) as Box<dyn Error>)
    }
}

/// Find model
/// Look for the model in the working directory and then in the binary's directory.
pub fn resolve_model_path(model_path: &str) -> Result<String, Box<dyn Error>> {
    if std::path::Path::new(model_path).exists() {
        Ok(model_path.into())
    } else {
        let mut exe = std::env::current_exe()?;
        exe.set_file_name(DEFAULT_MODEL);
        if !exe.exists() {
            Err(format!("model {} not found", model_path).into())
        } else {
            info!("Use default model {:?}", exe);
            Ok(exe.to_str().unwrap().into())
        }
    }
}

fn img2base64(img: &image::DynamicImage, png: bool) -> String {
    let mut cursor = Cursor::new(Vec::new());
    if png {
        img.write_to(&mut cursor, image::ImageOutputFormat::Png)
            .unwrap();
    } else {
        img.write_to(&mut cursor, image::ImageOutputFormat::Jpeg(75))
            .unwrap();
    }
    base64::encode(&cursor.into_inner())
}

/// Calculate cobb angle in degrees
pub fn calc_angle(points: &[Point]) -> f32 {
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
    let angle_degree = c2_line.vector.angle_to(c7_line.vector).to_degrees();
    if points[0].0 > points[1].0 {
        // facing right
        -angle_degree
    } else {
        // facing left
        angle_degree
    }
}

/// Draw lines and angle as SVG
pub fn draw(
    json_data: PointData,
    background: Option<image::DynamicImage>,
    png_bg: bool,
) -> svg::Document {
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
        .set("viewBox", (0i64, 0i64, image_width, image_height))
        .set("xmlns:xlink", "http://www.w3.org/1999/xlink");

    // set background
    if let Some(base_img) = background {
        let res_base64 = img2base64(&base_img, png_bg);
        let b64 =
            format!("data:image/{};base64,", if png_bg { "png" } else { "jpeg" }) + &res_base64;
        let bg = element::Image::new()
            .set("x", 0i64)
            .set("y", 0i64)
            .set("width", image_width)
            .set("height", image_height)
            .set("xlink:href", b64);
        document = document.add(bg);
    }

    let line_width = 2i32;
    let line_color = "yellow";
    let font_size = "24";
    let font_family = "sans-serif";
    let point_radius = "3";

    // draw lines
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
        let (int1, int2) = if (v.y / v.x).abs() > 0.5 {
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
    let angle_degree = calc_angle(&points);
    let angle_degree = if points[0].0 > points[1].0 {
        // facing right
        -angle_degree
    } else {
        // facing left
        angle_degree
    };
    let intersect = c2_line.intersection(&c7_line);
    let intersect = match intersect {
        // When c2_line and c7_line are parallel, use the middle point of p1 and p3 as a pseudo intersection
        None => c2_line.point.lerp(c7_line.point, 0.5),
        Some(p) => p,
    };
    let angle = format!("{:.1}°", angle_degree);
    debug!("angle {}", angle);

    let mut angle_text_position = intersect;

    if lyon_geom::Box2D::new(tl, br).contains(intersect) {
        debug!("No aux lines");
    } else {
        debug!("Draw aux lines");
        let a = lyon_geom::Point::new(points[0].0, points[0].1);
        let p = lyon_geom::Point::new(points[1].0, points[1].1);
        let aux_c2 = if a.distance_to(intersect) > p.distance_to(intersect) {
            p + (p - a)
        } else {
            a + (a - p)
        };
        let t_origin = lyon_geom::Transform::translation(-intersect.x, -intersect.y);
        let mut rot_angle = c2_line.vector.angle_to(c7_line.vector);
        rot_angle.radians /= 2.0;
        let t = t_origin
            .then_rotate(rot_angle)
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
            let f2 = if rot_angle.radians > 0.0 { 1 } else { 0 } as i32;
            data = data.elliptical_arc_by((radius, radius, slope, f1, f2, end.x, end.y));
            let path = element::Path::new().set("d", data);
            group = group.add(path);
        }

        angle_text_position = aux_int;
    }

    document = document.add(group);

    let text_attr = [
        ("font-family", font_family),
        ("font-size", font_size),
        ("fill", "black"),
        ("stroke", "white"),
        ("stroke-width", "0.5px"),
        ("font-weight", "bold"),
    ];

    // adjust angle text position when default position is too right
    let est_text_width = 0.8f32 * font_size.parse::<f32>().unwrap() * (angle.len() - 2) as f32;
    if angle_text_position.x + est_text_width > image_width as f32 {
        let a = lyon_geom::Point::new(points[0].0, points[0].1);
        let p = lyon_geom::Point::new(points[1].0, points[1].1);
        let c2dir = (p - a).abs().normalize();
        use core::ops::Mul;
        let text_init = if lyon_geom::Box2D::new(tl, br).contains(intersect) {
            intersect - c2dir.mul(est_text_width)
        } else {
            let aux_c2 = if a.distance_to(intersect) > p.distance_to(intersect) {
                p + (p - a)
            } else {
                a + (a - p)
            };
            aux_c2 - c2dir.mul(est_text_width)
        };
        let t_origin = lyon_geom::Transform::translation(-intersect.x, -intersect.y);
        let mut rot_angle = c2_line.vector.angle_to(c7_line.vector);
        rot_angle.radians /= 2.0;
        let t = t_origin
            .then_rotate(rot_angle)
            .then_translate(lyon_geom::Vector::new(intersect.x, intersect.y));
        let text_pos = t.transform_point(text_init);
        debug!(
            "Original and adjusted angle text positions. {:?}, {:?}",
            angle_text_position, text_pos
        );
        angle_text_position = text_pos;
    }

    let mut text = element::Text::new()
        .set("x", angle_text_position.x)
        .set("y", angle_text_position.y);
    text = text_attr
        .iter()
        .fold(text, |text, (attr, value)| text.set(*attr, *value));
    text = text.add(node::Text::new(angle));
    document = document.add(text);

    // draw points
    let colors = vec!["red", "lime", "blue", "magenta"];
    for (label, color) in LABELS.iter().zip(colors) {
        let point_xy = shape_map[*label];
        let circle = Circle::new()
            .set("fill", color)
            .set("stroke", "black")
            .set("stroke-width", 0)
            .set("cx", point_xy.0)
            .set("cy", point_xy.1)
            .set("r", point_radius);
        document = document.add(circle);
        // let mut text = element::Text::new()
        //     .set("x", point_xy.0)
        //     .set("y", point_xy.1);
        // text = text_attr
        //     .iter()
        //     .fold(text, |text, (attr, value)| text.set(*attr, *value));
        // text = text.add(node::Text::new(*label));
        // document = document.add(text);
    }
    document
}

/// Create a batch with mirrored inputs
pub fn create_input_batch(
    img: &image::ImageBuffer<image::Luma<u8>, Vec<u8>>,
    input_width: u32,
    input_height: u32,
) -> ndarray::Array4<f32> {
    ndarray::Array4::from_shape_fn(
        (2, 1, input_height as usize, input_width as usize),
        |(lr, c, y, x)| {
            if img.in_bounds(x as _, y as _) {
                if lr == 0 {
                    img[(x as _, y as _)][c] as f32 / 255.0
                } else {
                    // flip
                    img[((img.dimensions().0 - (x + 1) as u32) as _, y as _)][c] as f32 / 255.0
                }
            } else {
                0.0
            }
        },
    )
}

/// Extract landmark point out of the input heatmaps
pub fn extract_points(arr: &ndarray::Array3<u8>) -> Vec<Point> {
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
        let affinity = arr.slice(s![c2c7 + 4, .., ..]);
        debug!("Find optimal point pairs for {}", c2c7);
        let mut scores: Vec<(u8, Point, Point)> =
            Vec::with_capacity(cand_left.len() * cand_right.len());
        for left in cand_left.iter() {
            for right in cand_right.iter() {
                debug!("Calculate the score of {:?},{:?}", left, right);
                let l = (left.0 as isize, left.1 as isize);
                let r = (right.0 as isize, right.1 as isize);
                let score_n_count = line_drawing::Bresenham::new(l, r)
                    .map(|(y, x)| (affinity[[y as usize, x as usize]] as usize, 1))
                    .reduce(|accum, item| (accum.0 + item.0, accum.1 + item.1));
                let score = match score_n_count {
                    None => u8::MAX,
                    Some((sum, count)) => (sum as f64 / count as f64) as u8,
                };
                scores.push((score, *left, *right));
            }
        }
        let (optimal_score, left, right) = scores.into_iter().max_by_key(|t| t.0).unwrap();
        info!(
            "Optimal score for {}:{}",
            c2c7,
            optimal_score as f64 / u8::MAX as f64
        );
        optimal_points.push(left);
        optimal_points.push(right);
    }
    optimal_points.into_iter().map(|(y, x)| (x, y)).collect()
}

use dicom::dictionary_std;
use image::imageops::FilterType;
use image::{DynamicImage, GenericImageView, ImageBuffer};

pub fn load_dicom(filename: &str) -> Result<image::DynamicImage, Box<dyn Error>> {
    use std::fs::File;
    use std::io::Read;
    let mut f = File::open(filename)?;
    let mut bytes: Vec<u8> = Vec::new();
    f.read_to_end(&mut bytes)?;
    load_dicom_from_u8(&bytes)
}

///
/// # Arguments
/// - bytes: Raw bytes from a dicom file with preamble
pub fn load_dicom_from_u8(bytes: &[u8]) -> Result<image::DynamicImage, Box<dyn Error>> {
    assert!(bytes.len() > 128);
    let wo_preamble = &bytes[128..]; // skip preamble
    let obj = dicom::object::from_reader(wo_preamble)?;
    let pixel_representation: u8 = obj.element_by_name("PixelRepresentation")?.to_int()?;
    let bits_allocated: u8 = obj.element_by_name("BitsAllocated")?.to_int()?;
    let bits_stored: u8 = obj.element_by_name("BitsStored")?.to_int()?;
    debug!(
        "pixel representation {}, bits_stored {}",
        pixel_representation, bits_stored
    );
    // let compression: u8 = obj.element_by_name("LossyImageCompression")?.to_int()?;
    assert_eq!(bits_allocated, 16, "Not a 16bit dicom");
    let width: u32 = obj.element(dictionary_std::tags::COLUMNS)?.to_int()?;
    let height: u32 = obj.element(dictionary_std::tags::ROWS)?.to_int()?;

    let pixel_data_bytes = obj.element(dicom::core::Tag(0x7FE0, 0x0010))?.to_bytes()?;

    type PixelType = u16;
    let mut pixels: Vec<PixelType> =
        unsafe { (pixel_data_bytes.into_owned().align_to::<PixelType>().1).to_vec() };
    let monochrome = obj.element_by_name("PhotometricInterpretation")?.to_str()?;
    if monochrome == "MONOCHROME1" {
        info!("Invert pixel values");
        let pmax = *pixels.iter().max().unwrap();
        pixels.iter_mut().for_each(|p| *p = pmax - *p);
    }
    let pixels_u16: Vec<u16> = pixels.into_iter().collect();
    type OutputPixelType = u16;
    let buf: ImageBuffer<image::Luma<OutputPixelType>, Vec<OutputPixelType>> =
        ImageBuffer::from_raw(width, height, pixels_u16).unwrap();
    Ok(DynamicImage::ImageLuma16(buf))
}

/// Convert 16bit gray image into 8bit image using minmax normalization
pub fn luma16toluma8(img: &image::ImageBuffer<image::Luma<u16>, Vec<u16>>) -> image::DynamicImage {
    let max_value = *img.iter().max().unwrap() as f32;
    let buf: Vec<u8> = img
        .iter()
        .map(|p| (*p as f32 / max_value * 255.0).round() as u8)
        .collect();
    image::DynamicImage::ImageLuma8(
        image::ImageBuffer::from_vec(img.dimensions().0, img.dimensions().1, buf).unwrap(),
    )
}

/// Extract heatmap of landmark points as RGB image from 6 channel output.
pub fn extract_heatmap(arr: &ndarray::Array3<u8>) -> Result<image::RgbaImage, Box<dyn Error>> {
    let (chs, height, width) = (arr.shape()[0], arr.shape()[1], arr.shape()[2]);
    if chs != 0 {
        return Err(ndarray::ShapeError::from_kind(ndarray::ErrorKind::IncompatibleShape).into());
    }
    let channel_last_shape = (height, width, chs);
    let channel_last: ndarray::Array3<u8> = ndarray::Array3::from_shape_vec(
        channel_last_shape,
        arr.view()
            .permuted_axes([1, 2, 0])
            .iter()
            .cloned()
            .collect(),
    )?;
    let rows: Vec<u8> = channel_last
        .to_shape((width * height, 6))?
        .rows()
        .into_iter()
        .flat_map(|row| {
            let (mut r, g, mut b) = (row[0], row[1], row[2]);
            r += row[3];
            b += row[3];
            let alpha = u8::max(u8::max(r, g), b);
            [r, g, b, alpha]
        })
        .collect();
    image::RgbaImage::from_raw(width as _, height as _, rows)
        .ok_or_else(|| "Failed to convert to RGBA image".into())
}

/// Extract affinity map as RGB image from 6 channel output.
pub fn extract_affinity(arr: &ndarray::Array3<u8>) -> Result<image::RgbaImage, Box<dyn Error>> {
    let (chs, height, width) = (arr.shape()[0], arr.shape()[1], arr.shape()[2]);
    if chs != 0 {
        return Err(ndarray::ShapeError::from_kind(ndarray::ErrorKind::IncompatibleShape).into());
    }
    let channel_last_shape = (height, width, chs);
    let channel_last: ndarray::Array3<u8> = ndarray::Array3::from_shape_vec(
        channel_last_shape,
        arr.view()
            .permuted_axes([1, 2, 0])
            .iter()
            .cloned()
            .collect(),
    )?;
    let rows: Vec<u8> = channel_last
        .to_shape((width * height, 6))?
        .rows()
        .into_iter()
        .flat_map(|row| {
            let (r, g) = (row[4], row[5]);
            let alpha = u8::max(r, g);
            [r, g, 0, alpha]
        })
        .collect();
    image::RgbaImage::from_raw(width as _, height as _, rows)
        .ok_or_else(|| "Failed to convert to RGBA image".into())
}

/// Return the index of the best batch
/// Best batch is the batch with the largest min score
///
/// # Arguments
/// - arr4: shape of (2, 4, height, width)
pub fn choose_best_batch<S>(arr4: &ndarray::ArrayBase<S, ndarray::Ix4>) -> usize
where
    S: ndarray::Data<Elem = f32>,
{
    assert_eq!(arr4.shape()[0], 2);
    assert_eq!(arr4.shape()[1], 6);
    let v_maxes: Vec<Vec<f32>> = arr4
        .axis_iter(ndarray::Axis(0))
        .map(|output3| {
            let maxes: Vec<f32> = output3
                .axis_iter(ndarray::Axis(0))
                .map(|output2| output2.fold(f32::MIN, |acc, v| f32::max(acc, *v)))
                .collect();
            maxes
        })
        .collect();
    debug!("All scores {:?}", v_maxes);
    let mins: Vec<f32> = v_maxes
        .iter()
        .map(|v| v.iter().fold(f32::MAX, |acc, v| f32::min(acc, *v)))
        .collect();
    debug!("Scores per batch {:?}", mins);
    assert!(mins.len() == 2);
    let bb = if mins[0] > mins[1] { 0 } else { 1 };
    info!("Facing {}", if bb == 0 { "left" } else { "right" });
    bb
}

extern crate wasm_bindgen;
use wasm_bindgen::prelude::*;
extern crate console_error_panic_hook;
use std::panic;

#[wasm_bindgen(start)]
pub fn main() {
    panic::set_hook(Box::new(console_error_panic_hook::hook));
    wasm_logger::init(wasm_logger::Config::default());
    debug!("logger initialized");
}

#[wasm_bindgen(getter_with_clone)]
pub struct ImageB64 {
    pub b64: String,
    pub width: u32,
    pub height: u32,
}

/// load dicom or png/jpeg image
pub fn load_image(encoded: &[u8]) -> Result<image::DynamicImage, JsValue> {
    let dcm_img = load_dicom_from_u8(encoded);
    match dcm_img {
        Ok(img) => {
            info!("Dicom has been loaded");
            Ok(img)
        }
        Err(e) => {
            debug!("{}", e);
            let img = image::load_from_memory(encoded).map_err(|e| JsValue::from(e.to_string()))?;
            Ok(DynamicImage::ImageLuma8(img.to_luma8()))
        }
    }
}

#[wasm_bindgen]
pub fn decode_image(encoded: &[u8]) -> Result<ImageB64, JsValue> {
    let img = load_image(encoded)?;
    let img = match img {
        DynamicImage::ImageLuma8(img) => Some(DynamicImage::ImageLuma8(img)),
        DynamicImage::ImageLuma16(img) => Some(luma16toluma8(&img)),
        _ => Some(DynamicImage::ImageLuma8(img.to_luma8())),
    }
    .unwrap();
    let (width, height) = img.dimensions();
    let b64jpg = img2base64(&img, false);
    Ok(ImageB64 {
        b64: format!("data:image/jpeg;base64,{}", b64jpg),
        width,
        height,
    })
}

pub const TARGET_HEIGHT: usize = 768;

fn calc_resized_width(original_width: u32, original_height: u32) -> u32 {
    ((original_width as f64) * (TARGET_HEIGHT as f64) / (original_height as f64)).round() as u32
}

/// Calculate image width (tensor width) after resizing and padding
///
/// # Arguments
/// - image_width: Original (before resizing and padding) width.
#[wasm_bindgen]
pub fn calc_tensor_width(image_width: u32, image_height: u32) -> u32 {
    let resized_width: u32 =
        ((image_width as f64) * (TARGET_HEIGHT as f64) / (image_height as f64)).round() as u32;
    let new_width: u32 = ((resized_width as f64 / 256.0).ceil() * 256.0) as u32;
    new_width
}

#[wasm_bindgen]
pub fn create_input_tensor(encoded: &[u8]) -> Result<js_sys::Float32Array, JsValue> {
    let original_img = load_image(encoded)?;
    let (width, height) = original_img.dimensions();
    let gray_img = original_img.grayscale();

    let new_width: u32 = calc_resized_width(width, height);
    debug!("Original image size: {} {}", width, height);
    info!("Resize to w{} h{}", new_width, TARGET_HEIGHT);
    let resized_img = gray_img.resize_exact(new_width, TARGET_HEIGHT as _, FilterType::Triangle);
    let img = match &resized_img {
        DynamicImage::ImageLuma8(img) => {
            debug!("input u8 to clahe");
            clahe::clahe(img, 32, 32, 10)
        }
        DynamicImage::ImageLuma16(img) => {
            debug!("input u16 to clahe");
            clahe::clahe(img, 32, 32, 100)
        }
        _ => {
            return Err(JsValue::from("Unsupported image"));
        }
    }
    .map_err(|e| JsValue::from(e.to_string()))?;
    info!("Done clahe");
    let pad_width = calc_tensor_width(original_img.dimensions().0, original_img.dimensions().1);
    let (input_width, input_height) = (pad_width, TARGET_HEIGHT as _);

    info!("Create input tensor");
    let input_tensor: ndarray::Array4<f32> = create_input_batch(&img, input_width, input_height);
    debug!("Input shape {:?}", input_tensor.shape());
    let v = input_tensor.into_raw_vec();
    Ok(js_sys::Float32Array::from(&v[..]))
}

#[wasm_bindgen]
pub fn process_output(
    encoded: &[u8],
    raw_output: &[f32],
    tensor_width: u32,
    original_width: u32,
    original_height: u32,
) -> Result<String, JsValue> {
    let arr4 =
        ndarray::ArrayView4::from_shape((2, 6, TARGET_HEIGHT, tensor_width as _), raw_output)
            .map_err(|e| JsValue::from(e.to_string()))?;
    let best_batch: usize = choose_best_batch(&arr4) as _;

    let img_height = TARGET_HEIGHT;
    let img_width = calc_resized_width(original_width, original_height);
    let arr3 = arr4
        .slice(ndarray::s![
            best_batch,
            ..,
            0..img_height as usize,
            0..img_width as usize
        ])
        .map(|v| (v * 255.0) as u8);

    let flip_needed = best_batch != 0;
    let optimal_points = extract_points(&arr3);
    let optimal_points = if flip_needed {
        optimal_points
            .into_iter()
            .map(|(x, y)| (img_width as f32 - x - 1.0, y))
            .collect()
    } else {
        optimal_points
    };

    debug!("Optimal points {:?}", optimal_points);
    let labels: Vec<String> = LABELS.iter().map(|s| String::from(*s)).collect();
    let data = PointData::new(
        &optimal_points,
        &labels,
        img_width as _,
        img_height as _,
        "",
    );

    let background = load_image(encoded)?;
    let background = match background {
        DynamicImage::ImageLuma8(img) => Some(DynamicImage::ImageLuma8(img)),
        DynamicImage::ImageLuma16(img) => Some(luma16toluma8(&img)),
        _ => Some(DynamicImage::ImageLuma8(background.to_luma8())),
    }
    .unwrap();
    let document = draw(data, Some(background), false);
    Ok(document.to_string())
}

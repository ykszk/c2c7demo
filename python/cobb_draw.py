import sys
import argparse
import os
import json

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import sympy
from sympy.geometry import Point, Line, Polygon, Segment, Circle, intersection
from logzero import logger

colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 0, 255)]
fontname = 'arial.ttf' if os.name == 'nt' else 'DejaVuSans.ttf'


def draw_cobb(img, data):
    '''
    Args:
        img: PIL.Image
        data: dictionary of labelme. containing 4 points.
    '''

    draw = ImageDraw.Draw(img)

    def draw_circle(draw, center, radius, fill):
        center = np.array(center)
        xy = [tuple(center - radius), tuple(center + radius)]
        draw.ellipse(xy, fill=fill)

    kps = np.array([
        shape['points'][0] for shape in data['shapes']
        if shape['shape_type'] == 'point'
    ])
    width = (np.linalg.norm(kps[1] - kps[0]) +
             np.linalg.norm(kps[3] - kps[2])) / 2
    logger.debug('Vertebra width:%s', width)

    # draw lines
    line_color = (255, 255, 0)
    line_width = np.clip(int(width / 30), 2, None)
    logger.debug('Line width:%d', line_width)
    line_left = Line((0, 0), (0, img.size[1]))
    line_right = Line((img.size[0], 0), img.size)
    c2 = Line(kps[0], kps[1])
    c7 = Line(kps[2], kps[3])
    line_front = Segment(kps[0], kps[2])

    for c_line in [c2, c7]:
        left = tuple(intersection(c_line, line_left)[0].evalf())
        right = tuple(intersection(c_line, line_right)[0].evalf())
        draw.line([left, right], width=line_width, fill=line_color)

    # draw points
    for kp, color in zip(kps, colors):
        draw_circle(draw, kp, width / 20, color)
        # draw.arc(, 0, 360, fill=255)

    # draw angle
    font_color = (255, 0, 0)
    sign = 1 if float(
        Segment(kps[0], kps[1]).angle_between(
            Segment(kps[0],
                    intersection(c2, c7)[0]))) != 0 else -1
    angle = sign * np.rad2deg(float(c2.angle_between(c7)))
    poly_image = Polygon((0, 0), (img.size[0], 0), img.size, (0, img.size[1]))
    cross = intersection(c2, c7)[0]
    inside = poly_image.encloses_point(cross)
    font_size = int(np.clip(6 * width / 20, 12, None))
    logger.debug('Font size:%d', font_size)
    font = ImageFont.truetype(fontname, font_size)
    if inside:
        s_dir = sympy.Matrix((line_front.midpoint - cross).evalf())
        s_dir = s_dir / s_dir.norm(ord=2)
        text_pos = intersection(
            c2, c7)[0] - (font_size * 2, font_size / 2) + 2 * font_size * s_dir
        draw.text(text_pos,
                  '{}°'.format(np.round(angle, 1)),
                  font=font,
                  fill=font_color)
    else:
        aux_c2 = ((kps[0] + kps[1]) / 2 - 2 * sign * (kps[1] - kps[0]))
        aux_c7 = (kps[2] + kps[3]) / 2 - 2 * sign * (kps[3] - kps[2])
        aux_middle = (aux_c2 + aux_c7) / 2
        perpline_c2 = c2.perpendicular_line(aux_middle)
        perp_c2 = perpline_c2.intersection(c2)[0]
        perp_c2c7 = perpline_c2.intersection(c7)[0]
        perpline_c7 = c7.perpendicular_line(aux_middle)
        perp_c7 = perpline_c7.intersection(c7)[0]
        perp_c7c2 = c7.perpendicular_line(aux_middle).intersection(c2)[0]
        quantile = .75
        draw.line([
            tuple(perp_c2.evalf()),
            tuple((perp_c2 + (quantile * (perp_c2c7 - perp_c2))).evalf())
        ],
                  width=line_width,
                  fill=line_color)
        arc_radius = float(
            sympy.Matrix(perpline_c2.intersection(c7)[0] -
                         perp_c2).norm(ord=2).evalf()) * .5 * (1 - quantile)
        aux_circle = Circle(aux_middle, arc_radius)

        def p2deg(p):
            return np.rad2deg(float(sympy.atan2(p[1], p[0]).evalf()))

        def draw_arc(draw, center, radius, start, end, fill, width):
            center = np.array(center)
            xy = [tuple(center - radius), tuple(center + radius)]
            draw.arc(xy, start, end, fill=fill, width=width)

        draw.line([
            tuple(perp_c7.evalf()),
            tuple((perp_c7 + (quantile * (perp_c7c2 - perp_c7))).evalf())
        ],
                  width=line_width,
                  fill=line_color)

        theta1 = p2deg(
            aux_circle.intersection(Segment(aux_middle, perp_c2))[0] -
            aux_middle)
        theta2 = p2deg(
            aux_circle.intersection(Segment(aux_middle, perp_c7c2))[0] -
            aux_middle)
        theta1, theta2 = sorted([theta1, theta2])
        draw_arc(draw,
                 aux_middle,
                 arc_radius,
                 theta1,
                 theta2,
                 fill=line_color,
                 width=line_width)

        theta1 = p2deg(
            aux_circle.intersection(Segment(aux_middle, perp_c7))[0] -
            aux_middle)
        theta2 = p2deg(
            aux_circle.intersection(Segment(aux_middle, perp_c2c7))[0] -
            aux_middle)
        theta1, theta2 = sorted([theta1, theta2], reverse=True)
        draw_arc(draw,
                 aux_middle,
                 arc_radius,
                 theta2,
                 theta1,
                 fill=line_color,
                 width=line_width)

        s_dir = sympy.Matrix((line_front.midpoint - aux_middle).evalf())
        s_dir = s_dir / s_dir.norm(ord=2)
        text_pos = Point(aux_middle) - (font_size * 2,
                                        font_size / 2) + 2 * font_size * s_dir
        draw.text(text_pos,
                  '{}°'.format(np.round(angle, 1)),
                  font=font,
                  fill=font_color)


def main():
    parser = argparse.ArgumentParser(description='Visualize C2-C7 Cobb angle.')
    parser.add_argument('input',
                        help='Input image filename:',
                        metavar='<input>')
    parser.add_argument('json', help='Input json filename:', metavar='<json>')
    parser.add_argument('output',
                        help='Output image filename',
                        metavar='<output>')
    parser.add_argument('-o',
                        '--option',
                        help='Optional argument. default: %(default)s',
                        metavar='<name>',
                        default='default')
    parser.add_argument('-v',
                        '--verbose',
                        help='Verbose mode.',
                        action='store_true')

    args = parser.parse_args()

    if args.verbose:
        logger.setLevel('DEBUG')
    else:
        logger.setLevel('INFO')

    with open(args.json) as f:
        data = json.load(f)

    img = Image.open(args.input).convert('RGB')
    draw_cobb(img, data)
    img.save(args.output)


if __name__ == '__main__':
    sys.exit(main())

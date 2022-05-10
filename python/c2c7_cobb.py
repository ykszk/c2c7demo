import sys
import argparse
from pathlib import Path
import json

import numpy as np
import skimage
import skimage.io
import skimage.transform
import torch
import cobb_draw
from PIL import Image
from logzero import logger
from utils import (create_model, load_image, resize_height,
                   normalize_intensity, pad_to_shape, apply_clahe,
                   base_transform, heatmap2rgb, heatmap2points, rgb_on_gray)
import utils

KP_LABELS = ['C2A', 'C2P', 'C7A', 'C7P']


def main():
    parser = argparse.ArgumentParser(description='C2-C7 angle.')
    parser.add_argument('input', help='Input filename:', metavar='<input>')
    parser.add_argument('output', help='Output filename', metavar='<output>')
    parser.add_argument('-w',
                        '--weights',
                        help='pt file path.',
                        metavar='<filename>',
                        default='weights/c2c7_ENetB4.pth')
    parser.add_argument('--json',
                        help='Output labelme json.',
                        metavar='<filename>')
    parser.add_argument('--pped',
                        help='Output pre-processed image.',
                        metavar='<filename>')
    parser.add_argument('--npy',
                        help='Output heatmap as np.array. (e.g. heatmap.npy)',
                        metavar='<filename>')
    parser.add_argument('--heatmap',
                        help='Output heatmap image. (e.g. heatmap.jpg)',
                        metavar='<filename>')
    parser.add_argument('--affinity',
                        help='Output affinity image. (e.g. affinity.jpg)',
                        metavar='<filename>')
    parser.add_argument('--flip',
                        help='Horizontally flip input image.',
                        action='store_true')
    parser.add_argument('--height',
                        help='Image height.',
                        default=768,
                        type=int,
                        metavar='<height>')
    parser.add_argument('-v',
                        '--verbose',
                        help='Verbose mode.',
                        action='store_true')

    args = parser.parse_args()
    if args.verbose:
        logger.setLevel('DEBUG')
    else:
        logger.setLevel('INFO')

    print('Load', args.input, end=' ')
    original_img = load_image(args.input)
    if args.flip:
        original_img = np.flip(original_img, axis=1)
    print('done.')
    original_shape = original_img.shape
    logger.debug('Original shape %s', original_shape)
    img = resize_height(args.height, original_img)
    resized_shape = img.shape
    if original_img.dtype != np.uint8:
        original_img = normalize_intensity(original_img)
    down_factor = 256
    new_shape = (down_factor *
                 np.ceil(np.array(img.shape) / down_factor)).astype(np.int64)
    img = pad_to_shape(img, new_shape)
    padded_shape = img.shape
    logger.debug('Input shape %s', padded_shape)

    img = apply_clahe(img)
    if args.pped:
        print('Save pre-processed image: ', args.pped, end=' ')
        skimage.io.imsave(args.pped, img)
        print('done')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('Load model', end=' ')
    model = create_model()
    model.load_state_dict(torch.load(args.weights))
    print('done.')
    model.to(device)
    model.eval()
    x = base_transform(img)
    x = torch.from_numpy(x[np.newaxis]).to(device)
    print('Calculate heatmaps', end=' ')
    with torch.no_grad():
        y = model(x).to('cpu').detach().numpy().copy()
        if args.flip:
            y = np.flip(y, axis=-1)
            original_img = np.flip(original_img, axis=1)
    print('done.')
    y = y.squeeze()
    if args.flip:
        y = y[:, (y.shape[1] - resized_shape[0]):, (y.shape[2] - resized_shape[1]):]  # remove paddings
    else:
        y = y[:, :resized_shape[0], :resized_shape[1]]  # remove paddings
    y = np.clip(np.round(y.squeeze() * 255), 0, 255).astype(np.uint8)

    if args.npy:
        print('Save raw heatmap: ', args.npy, end=' ')
        np.save(args.npy, y)
        print('done')
    if args.heatmap:
        print('Save heatmap: ', args.heatmap, end=' ')
        rgb = heatmap2rgb(
            y[:4],
            [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1]],
        )
        rgb = skimage.transform.resize(rgb,
                                       original_shape,
                                       preserve_range=True)
        overlay = rgb_on_gray(rgb, original_img)
        skimage.io.imsave(args.heatmap, overlay, check_contrast=False)
        print('done')
    if args.affinity:
        print('Save affinity map: ', args.affinity, end=' ')
        rgb = heatmap2rgb(
            y[4:],
            [[1, 0, 0], [0, 1, 0]],
        )
        rgb = skimage.transform.resize(rgb,
                                       original_shape,
                                       preserve_range=True)
        overlay = rgb_on_gray(rgb, original_img)
        skimage.io.imsave(args.affinity, overlay, check_contrast=False)
        print('done')

    kps, scores = heatmap2points(y)
    kps = np.array(kps)
    logger.debug('Scores:%s', scores)
    kps = (original_shape[0] / resized_shape[0]) * kps
    logger.debug('Key points: %s', kps)

    json_data = {
        "version": "4.5.7",
        "flags": {},
        "shapes": [],
        "imagePath": Path(args.input).name,
        "imageData": None,
        "imageHeight": original_shape[0],
        "imageWidth": original_shape[1]
    }
    shapes = json_data['shapes']
    for kp, label in zip(kps, KP_LABELS):
        shapes.append({
            "label": label,
            "points": [kp.tolist()],
            "group_id": None,
            "shape_type": "point",
            "flags": {}
        })

    if args.json:
        print('Save json: ', args.json, end=' ')
        with open(args.json, 'w') as f:
            json.dump(json_data, f, indent=2)
        print(' done.')

    print('Save ', args.output, end=' ')
    pil = Image.fromarray(original_img).convert('RGB')
    cobb_draw.draw_cobb(pil, json_data)
    pil.save(args.output)
    print('done.')

    return 0


if __name__ == '__main__':
    sys.exit(main())
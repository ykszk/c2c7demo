from pathlib import Path
import itertools

import numpy as np
import skimage
import skimage.transform
from skimage import measure
import segmentation_models_pytorch as smp
import cv2
import pydicom
from scipy.ndimage.measurements import center_of_mass
import torch
from torch import nn

def create_model():
    model = smp.DeepLabV3Plus(encoder_name='efficientnet-b4',
                              encoder_weights=None,
                              activation=None,
                              in_channels=1,
                              classes=6)
    return model

def apply_clahe(img):
    clahe = cv2.createCLAHE(clipLimit=100, tileGridSize=(32, 32))
    cl = clahe.apply(img)
    cl = cl - cl.min()
    cl = cl / cl.max()
    return np.round(np.clip(cl, 0, 1) * 255).astype(np.uint8)


def resize_height(height, img):
    '''
    Resize image to the height. Aspect ratio is kept.
    Args:
        height: target image height
        img (ndarray): img
    '''
    dtype = img.dtype
    assert dtype == np.uint8 or dtype == np.uint16, 'Invalid image dtype'
    scale = height / img.shape[0]
    width = img.shape[1]
    width = np.round(scale * width).astype(int)
    img = skimage.transform.resize(img, (height, width), preserve_range=True)
    img = np.round(img).astype(dtype)
    return img


def load_dicom(filename):
    dcm = pydicom.dcmread(filename)
    sign = 1 if dcm.PhotometricInterpretation == 'MONOCHROME2' else -1
    img = sign * dcm.pixel_array
    if hasattr(dcm, 'RescaleSlope') and hasattr(dcm, 'RescaleIntercept'):
        img = dcm.RescaleSlope * img + dcm.RescaleIntercept
    return img


def load_image(filename):
    filename = Path(filename)
    if filename.suffix == '' or filename.suffix == '.dcm':
        return load_dicom(filename)
    else:
        img = skimage.io.imread(filename)
        if img.ndim == 3:
            img = img[..., 0]
        return img


def calc_candidates(heatmap):
    all_candidates = []
    for ch in heatmap[:4]:
        max_value = np.max(ch)
        if max_value <= 0:
            raise RuntimeError('Empty heatmap')
        thresh = max_value / 2
        all_labels = measure.label(ch >= thresh)
        candidates = []
        for i in range(1, np.max(all_labels) + 1):
            cc = (all_labels == i)
            center = center_of_mass(ch * (cc == cc.max()))
            candidates.append(center[::-1])
        all_candidates.append(candidates)
    return all_candidates


def heatmap2points(heatmap):
    all_candidates = calc_candidates(heatmap)

    keypoints = []
    best_scores = []
    for i, ch in enumerate(heatmap[4:]):
        c1s, c2s = all_candidates[2 * i], all_candidates[2 * i + 1]
        if len(c1s) == 0 and len(c2s) == 0:
            pair = [c1s[0], c2s[0]]
        else:
            scores, pairs = [], []
            for c1, c2 in itertools.product(c1s, c2s):
                line = np.zeros_like(ch)
                ic1, ic2 = tuple(np.round(c1).astype(int)), tuple(
                    np.round(c2).astype(int))
                line = cv2.line(line, ic2, ic1, 1, 1)
                line_sum = np.sum(ch * line) / np.sum(line)
                scores.append(line_sum)
                pairs.append((c1, c2))
            best_scores.append(np.max(scores))
            pair = pairs[np.argmax(scores)]
        keypoints.extend(pair)
    return keypoints, best_scores


def base_transform(x):
    if x.dtype == np.uint8:
        x = x / 255
    if x.ndim == 2:
        x = x[np.newaxis]
    x = x.astype(np.float32)
    return x


def pad_to_shape(x, shape):
    '''
    Add zero padding to bottom right.
    '''
    ds = [ns - os for ns, os in zip(shape, x.shape)]
    return np.pad(x, [(0, d) for d in ds])


def heatmap2rgb(heatmap, colormap):
    colormap = np.array(colormap).astype(np.float32)
    rgb = heatmap[..., np.newaxis] * colormap.reshape((len(colormap), 1, 1, 3))
    rgb = np.clip(np.round(rgb.sum(axis=0)), 0, 255).astype(np.uint8)
    return rgb


def normalize_intensity(x, percentile=(1, 99)):
    minmax = np.percentile(x, percentile)
    x = np.clip(x, minmax[0], minmax[1]).astype(np.float32)
    return np.round(255 * (x - minmax[0]) / (minmax[1] - minmax[0])).astype(
        np.uint8)

def rgb_on_gray(rgb, gray):
    '''
    overlay
    '''
    opacity = np.max(rgb, axis=-1)[..., np.newaxis] / 255.0
    overlay = np.repeat(gray[..., np.newaxis], 3,
                        -1) * (1 - opacity) + rgb * opacity
    overlay = np.round(overlay).astype(np.uint8)
    return overlay
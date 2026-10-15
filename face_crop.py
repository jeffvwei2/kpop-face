"""
Face detection and cropping shared by api.py (inference) and
curation_agent.py (building training data), using MediaPipe's BlazeFace
short-range model in models/.
"""

import os

import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import FaceDetector, FaceDetectorOptions

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

_detector = None


def detect_faces(pil_img):
    """All face detections in a PIL RGB image, highest score first."""
    global _detector
    if _detector is None:  # loaded lazily so importing this module stays cheap
        _detector = FaceDetector.create_from_options(FaceDetectorOptions(
            base_options=BaseOptions(
                model_asset_path=os.path.join(BASE_DIR, 'models', 'blaze_face_short_range.tflite')
            )
        ))
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.array(pil_img))
    detections = _detector.detect(mp_image).detections
    return sorted(detections, key=lambda d: d.categories[0].score, reverse=True)


def crop_box(pil_img, bbox, pad_ratio=0.25, square=False):
    """Crop around a detection's bounding box with padding on each side.
    square=True expands to a square centered on the face (clamped to the
    image), matching the framing of the existing training images."""
    x, y, w, h = bbox.origin_x, bbox.origin_y, bbox.width, bbox.height
    W, H = pil_img.size
    if square:
        side = max(w, h) * (1 + 2 * pad_ratio)
        cx, cy = x + w / 2, y + h / 2
        left, top = cx - side / 2, cy - side / 2
        right, bottom = left + side, top + side
    else:
        left, top = x - w * pad_ratio, y - h * pad_ratio
        right, bottom = x + w + w * pad_ratio, y + h + h * pad_ratio
    return pil_img.crop((max(0, left), max(0, top), min(W, right), min(H, bottom)))


def crop_to_face(pil_img, pad_ratio=0.25):
    """Crop to the most confident face, or None if no face is found."""
    detections = detect_faces(pil_img)
    if not detections:
        return None
    return crop_box(pil_img, detections[0].bounding_box, pad_ratio)

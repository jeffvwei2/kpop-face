import io
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import pandas as pd
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import FaceDetector, FaceDetectorOptions
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
from KpopClassifier import KpopClassifier

app = Flask(__name__)
CORS(app)

device = torch.device("cpu")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

train_df = pd.read_csv(os.path.join(BASE_DIR, 'data', 'combined_train.csv'))
test_df  = pd.read_csv(os.path.join(BASE_DIR, 'data', 'combined_test.csv'))
available_class = sorted(set(train_df['name'].unique()) & set(test_df['name'].unique()))
num_classes = len(available_class)
print(f"Loaded {num_classes} classes")


class ArcFaceLoss(nn.Module):
    def __init__(self, num_classes, embedding_dim=512):
        super().__init__()
        self.scale  = 64
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, embedding_dim))
        nn.init.xavier_uniform_(self.weight)


model     = KpopClassifier(num_classes=num_classes, embedding_dim=512, use_classifier_head=False).to(device)
criterion = ArcFaceLoss(num_classes=num_classes).to(device)

checkpoint = torch.load(os.path.join(BASE_DIR, 'best_model.pth'), map_location=device)
model.load_state_dict(KpopClassifier.state_dict_without_head(checkpoint['model']))
if checkpoint.get('criterion'):
    criterion.load_state_dict(checkpoint['criterion'])

model.eval()
criterion.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# detect the face first and crop to it (with some margin)
face_detector = FaceDetector.create_from_options(
    FaceDetectorOptions(
        base_options=BaseOptions(
            model_asset_path=os.path.join(BASE_DIR, 'models', 'blaze_face_short_range.tflite')
        )
    )
)


def crop_to_face(pil_img, pad_ratio=0.25):
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.array(pil_img))
    detections = face_detector.detect(mp_image).detections
    if not detections:
        return None

    best = max(detections, key=lambda d: d.categories[0].score)
    bbox = best.bounding_box
    x, y, w, h = bbox.origin_x, bbox.origin_y, bbox.width, bbox.height
    pad_x, pad_y = w * pad_ratio, h * pad_ratio
    W, H = pil_img.size
    left   = max(0, x - pad_x)
    top    = max(0, y - pad_y)
    right  = min(W, x + w + pad_x)
    bottom = min(H, y + h + pad_y)
    return pil_img.crop((left, top, right, bottom))


@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    img = Image.open(io.BytesIO(request.files['image'].read())).convert('RGB')

    face_img = crop_to_face(img)
    if face_img is None:
        return jsonify({'error': 'No face detected in image'}), 400

    tensor = transform(face_img).unsqueeze(0).to(device)

    with torch.no_grad():
        _, embeddings = model(tensor, return_embeddings=True)
        weight_norm   = F.normalize(criterion.weight, p=2, dim=1)
        logits        = torch.mm(embeddings, weight_norm.t()) * criterion.scale
        probs         = F.softmax(logits, dim=1)[0]

    top5 = torch.topk(probs, min(5, num_classes))
    results = [
        {'name': available_class[idx.item()], 'confidence': round(prob.item() * 100, 1)}
        for prob, idx in zip(top5.values, top5.indices)
    ]
    return jsonify({'predictions': results})


if __name__ == '__main__':
    app.run(port=5001, debug=False)

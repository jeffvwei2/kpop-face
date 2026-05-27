import io
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
from KpopClassifier import KpopClassifier

app = Flask(__name__)
CORS(app)

device = torch.device("cpu")

train_df = pd.read_csv('./data/kid_f_train.csv')
test_df  = pd.read_csv('./data/kid_f_test.csv')
available_class = sorted(set(train_df['name'].unique()) & set(test_df['name'].unique()))
num_classes = len(available_class)
print(f"Loaded {num_classes} classes")


class ArcFaceLoss(nn.Module):
    def __init__(self, num_classes, embedding_dim=512):
        super().__init__()
        self.scale  = 64
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, embedding_dim))
        nn.init.xavier_uniform_(self.weight)


model     = KpopClassifier(num_classes=num_classes, embedding_dim=512).to(device)
criterion = ArcFaceLoss(num_classes=num_classes).to(device)

checkpoint = torch.load('best_model.pth', map_location=device)
model.load_state_dict(checkpoint['model'])
if checkpoint.get('criterion'):
    criterion.load_state_dict(checkpoint['criterion'])

model.eval()
criterion.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    img    = Image.open(io.BytesIO(request.files['image'].read())).convert('RGB')
    tensor = transform(img).unsqueeze(0).to(device)

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

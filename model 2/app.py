from flask import Flask, request, render_template, redirect
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import os

app = Flask(__name__)

# Number of Decease goes here =>
num_classes = 10

# Loading... the trained model =>
model = models.resnet50(pretrained=False)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, num_classes)
model.load_state_dict(torch.load('models/skin_disease_model.pth', map_location=torch.device('cpu')))
model.eval()

# Defining transform =>
img_width, img_height = 150, 150
transform = transforms.Compose([
    transforms.Resize((img_width, img_height)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


def predict_skin_disease(image_path):
    img = Image.open(image_path).convert('RGB')
    img = transform(img).unsqueeze(0)
    with torch.no_grad():
        outputs = model(img)
        _, predicted = torch.max(outputs.data, 1)
    class_labels = {
        0: 'Eczema 1677',
        1: 'Melanoma',
        2: 'Atopic Dermatitis',
        3: 'Basal Cell Carcinoma (BCC)',
        4: 'Melanocytic Nevi (NV)',
        5: 'Benign Keratosis-like Lesions (BKL)',
        6: 'Psoriasis pictures Lichen Planus and related diseases',
        7: 'Seborrheic Keratoses and other Benign Tumors',
        8: 'Tinea Ringworm Candidiasis and other Fungal Infections',
        9: 'Warts Molluscum and other Viral Infections',
    }
    return class_labels[predicted.item()]


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file:
            file_path = os.path.join('uploads', file.filename)
            file.save(file_path)
            prediction = predict_skin_disease(file_path)
            return render_template('result.html', prediction=prediction)
    return render_template('upload.html')


if __name__ == '__main__':
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    app.run(debug=True)

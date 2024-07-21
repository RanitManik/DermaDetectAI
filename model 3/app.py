from flask import Flask, request, render_template, redirect
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import os

app = Flask(__name__)

# Number of Decease goes here =>
num_classes = 23

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
      0: 'Acne and Rosacea',
    1: 'Actinic Keratosis Basal Cell Carcinoma and other Malignant Lesions',
    2: 'Atopic Dermatitis',
    3: 'Bullous Disease',
    4: 'Cellulitis Impetigo and other Bacterial Infections',
    5: 'Eczema',
    6: 'Exanthems and Drug Eruptions',
    7: 'Hair Loss Alopecia and other Hair Diseases',
    8: 'Herpes HPV and other STDs',
    9: 'Light Diseases and Disorders of Pigmentation',
    10: 'Lupus and other Connective Tissue diseases',
    11: 'Melanoma Skin Cancer Nevi and Moles',
    12: 'Nail Fungus and other Nail Disease',
    13: 'Poison Ivy and other Contact Dermatitis',
    14: 'Psoriasis Lichen Planus and related diseases',
    15: 'Scabies Lyme Disease and other Infestations and Bites',
    16: 'Seborrheic Keratoses and other Benign Tumors',
    17: 'Systemic Disease',
    18: 'Tinea Ringworm Candidiasis and other Fungal Infections',
    19: 'Urticaria Hives',
    20: 'Vascular Tumors',
    21: 'Vasculitis',
    22: 'Warts Molluscum and other Viral Infections'
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

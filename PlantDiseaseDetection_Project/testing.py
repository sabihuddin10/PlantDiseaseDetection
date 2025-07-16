import os
import random
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights
from PIL import Image
import pandas as pd
from sklearn.preprocessing import LabelEncoder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(model_path, num_classes):
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model

def prepare_label_encoder(csv_path):
    df = pd.read_csv(csv_path)
    class_columns = df.columns[1:]
    df["labels"] = df[class_columns].apply(lambda row: class_columns[row.values == 1][0], axis=1)
    label_encoder = LabelEncoder()
    label_encoder.fit(df["labels"])
    return label_encoder

def predict_images(model_path, data_folder, num_images=5):
    image_dir = os.path.join(data_folder, "images")
    csv_path = os.path.join(data_folder, "train.csv")

    label_encoder = prepare_label_encoder(csv_path)
    model = load_model(model_path, num_classes=len(label_encoder.classes_))

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    image_files = [f for f in os.listdir(image_dir) if f.endswith(".jpg")]
    selected = random.sample(image_files, min(num_images, len(image_files)))

    for img_name in selected:
        img_path = os.path.join(image_dir, img_name)
        image = Image.open(img_path).convert("RGB")
        input_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(input_tensor)
            probs = torch.nn.functional.softmax(output, dim=1)
            conf, pred_idx = torch.max(probs, dim=1)
            pred_label = label_encoder.inverse_transform([pred_idx.item()])[0]

        print(f"Image: {img_name}")
        print(f"Predicted Label: {pred_label}")
        print(f"Confidence: {conf.item()*100:.2f}%")
        print("-" * 40)

# === USAGE ===
def main():
    model_path = r"D:\DLP_22k4347_22k4612_Project\resnet18_22k4347_22k4612_PlantDisease.pth"
    data_folder = os.path.join(os.getcwd(), "plant_data")
    predict_images(model_path, data_folder, num_images=295)
main()
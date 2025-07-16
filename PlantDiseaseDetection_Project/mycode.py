#lib imports
import os
import zipfile
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from PIL import Image
import os
import zipfile
from torchvision.models import resnet18, ResNet18_Weights


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device: ",device)

def prep_data(zip_path,extract_path):
  extract_path = os.path.join(os.getcwd(), "plant_data")
  if not os.path.exists(extract_path):
    with zipfile.ZipFile(zip_path,'r') as zip_ref:
      zip_ref.extractall(extract_path)
  print("Dataset has been extracted")
  return pd.read_csv(os.path.join(extract_path,"train.csv"))

def preprocess_dataframe(df):
  class_columns = df.columns[1:]
  df["labels"] = df[class_columns].apply(lambda row: class_columns[row.values == 1][0],axis=1)
  label_encoder = LabelEncoder()
  df["encoded"] = label_encoder.fit_transform(df["labels"])
  return df, label_encoder
def get_loaders( df , image_dir , transform ):
  train_df, val_df = train_test_split(df, test_size=0.2, stratify=df["encoded"], random_state=42)
  train_ds = PlantDataset(train_df, image_dir, transform)
  val_ds = PlantDataset(val_df, image_dir, transform)
  train_loader= DataLoader(train_ds, batch_size=32, shuffle=True)
  val_loader = DataLoader(val_ds, batch_size=32)
  return train_loader, val_loader

class PlantDataset(Dataset):
  def __init__(self,dataframe,image_dir,transform=None):
    self.df = dataframe
    self.image_dir = image_dir
    self.transform = transform
  def __len__(self):
    return len(self.df)
  def __getitem__(self,idx):
    img_id = self.df.iloc[idx]['image_id']
    label = self.df.iloc[idx]['encoded']
    img_path = os.path.join(self.image_dir,img_id + ".jpg")
    image = Image.open(img_path).convert("RGB")
    if self.transform:
      image = self.transform(image)
    return image, label
#Loaders

def save_model(model,path):
  torch.save(model.state_dict(),path)
  print(f"Model saved at {path}")
def main():
  model_name = "resnet18_22k4347_22k4612_PlantDisease.pth"
  model_path = os.path.join(os.getcwd(), model_name)
  zip_path = "plant-pathology-2020-fgvc7.zip"
  extract_path = os.path.join(os.getcwd(), "plant_data")
  image_dir = os.path.join(extract_path,"images")
  df = prep_data(zip_path,extract_path)
  df, label_encoder = preprocess_dataframe(df)
  #Transform
  transform = transforms.Compose([
      transforms.Resize((224, 224)),
      transforms.ToTensor(),
      transforms.Normalize([0.485, 0.456, 0.406],
                          [0.229, 0.224, 0.225])])
  
  train_loader, val_loader = get_loaders(df, image_dir,transform)
  model = resnet18(weights=ResNet18_Weights.DEFAULT)
  model.fc = nn.Linear(model.fc.in_features, len(label_encoder.classes_))
  model = model.to(device)
  criterion = nn.CrossEntropyLoss()
  optimizer = optim.Adam(model.parameters(),lr=0.0001)
  num_epochs = 4

  for epoch in range(num_epochs):
    model.train()
    total, correct, running_loss = 0, 0, 0.0
    for images, labels in train_loader:
      images, labels = images.to(device), labels.to(device)
      optimizer.zero_grad()
      outputs = model(images)
      loss = criterion(outputs,labels)
      loss.backward()
      optimizer.step()

      running_loss = running_loss + loss.item()
      _, preds = torch.max(outputs,1)
      total += labels.size(0)
      correct += (preds == labels).sum().item()
    acc = 100 * correct / total
    print("##################################")
    print(f"Epoch {epoch+1} / {num_epochs} : ")
    print(f"Loss : {running_loss:.4f} , Accuracy : {acc:.2f} %")
    print("##################################")
  save_model(model,model_path)
  print(f"Model saved at {model_path}")
if __name__ == "__main__":
  main()
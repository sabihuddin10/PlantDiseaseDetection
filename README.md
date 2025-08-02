# PlantDiseaseDetection
The repository contains a Plant Disease Detection System, built on a Custome ResNet Model
Here's your **complete `README.md`** as one single raw Markdown text file — ready to copy-paste directly into your GitHub project:

---

```markdown
# 🌿 Plant Disease Detection Using Deep Learning

This is an end-to-end plant disease classification system built using a **custom ResNet18 CNN model in PyTorch**. It predicts diseases from leaf images with high accuracy and supports both textual and visual inference.

> 🔬 Developed as part of my Deep Learning semester project (Spring 2025) — now extended as a personal project to explore model generalization, architecture tuning, and inference deployment.

---

## 📌 Objective

To automate identification of plant leaf diseases using deep learning, reducing reliance on manual inspection, which is often error-prone and time-consuming.

---

## 🏗️ Model Architecture

This project modifies the original ResNet18 as follows:

- Base: **Pretrained ResNet18** (ImageNet weights)
- Custom Head:
  - `Linear(in_features → 256)`
  - `ReLU`
  - `Dropout(p=0.3)`
  - `Linear(256 → num_classes)`
  - `Softmax` (applied at inference)

### 🔍 Visual Overview

```

Input Image (224x224x3)
→ ResNet18 Base Layers
→ AdaptiveAvgPool
→ Flatten
→ Linear → ReLU → Dropout
→ Final Linear → Softmax Output

```

---

## 📁 Dataset

- **Source:** Plant Pathology 2020 (Kaggle)
- **Classes:**
  - Healthy
  - Multiple Diseases
  - Rust
  - Scab
- **Preprocessing:**
  - Resize to 224×224
  - Normalize using ImageNet mean/std
  - Convert labels from one-hot → text → integer

---

## 🏋️ Training Configuration

| Component         | Value                  |
|------------------|------------------------|
| Model            | Custom ResNet18        |
| Optimizer        | Adam (lr = 0.0001)     |
| Loss Function    | CrossEntropyLoss       |
| Epochs           | 100                    |
| Batch Size       | 32                     |
| Device           | CUDA / CPU fallback    |
| Dropout          | 0.3                    |

Training logs and accuracy/loss plots are saved and visualized.

---

## 🧪 Results

- **Training Accuracy:** Fluctuated between 98% to 100%  
- **Training Loss:** Decreased steadily from 1.0 to ~0  
- **Inference Modes:**
  - `inference_print.py` – CLI prediction
  - `inference_visual.py` – matplotlib display with label

### ⚠️ Overfitting Observed
Due to dataset size and model capacity, the training accuracy became extremely high, but generalization is limited. This was a learning experience in model regularization and validation.

---

## 🔧 Future Improvements

We aim to reduce overfitting and improve generalization by:

- Adding more real-world leaf image data  
- Applying data augmentation (rotation, flips, crops, jitter)  
- Increasing dropout and adding L2 weight decay  
- Using early stopping and k-fold cross-validation  
- Scheduling learning rate decay  

---

## 🗂️ Project Structure

```

├── custom\_model.py           # ResNet18 customization
├── train\_model.py            # Training logic, saving model
├── inference\_print.py        # Text-based predictions
├── inference\_visual.py       # Image-based visual inference
├── training\_graph.png        # Accuracy/loss plot
├── requirements.txt
└── README.md

```

---

## 👤 Author

**Sabih Uddin Meraj**  
FAST NUCES Karachi  
Email: sabih.meraj@gmail.com  
LinkedIn: [linkedin.com/in/sabihmeraj](https://linkedin.com/in/sabihmeraj)  
GitHub: [github.com/sabih-uddin](https://github.com/sabih-uddin)

Special thanks to **Muhammad Umair Asad** for collaboration and **Sir Jawwad Shamsi** for mentorship.

---

## 📜 License

This project is open-sourced for educational and research use. Attribution appreciated.

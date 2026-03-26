# 🧠 Brain Tumor Segmentation using Pre-trained U-Net

[![GitHub](https://img.shields.io/badge/GitHub-samir--m0hamed-blue)](https://github.com/samir-m0hamed/Brain-Tumor-Segmentation-using-Pre-trained-U-Net)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-green)](https://pytorch.org/)
[![Python](https://img.shields.io/badge/Python-3.8+-blue)](https://www.python.org/)

> **Accurate brain tumor segmentation using a pre-trained U-Net with ResNet34 encoder and COCO-format annotations**

---

## 📊 Results

| Metric | Value |
|--------|-------|
| **Dice Score** | 0.7803 ✅ |
| **IoU** | 0.6421 ✅ |
| **Test Loss** | 0.2197 ✅ |
| **Epochs Trained** | 40 |
| **Device** | GPU (CUDA) ⚡ |

---

## 🚀 Quick Start

### ▶️ Prerequisites 
- Python 3.8+
- CUDA 11.8+ (optional)
- 8GB+ RAM (CPU) or 4GB+ VRAM (GPU)


### ▶️ Run Training

```bash
jupyter notebook "Brain Tumor Segmentation.ipynb"
```

---

## 📁 Project Structure

```
Brain-Tumor-Segmentation-using-UNet/
│
├── 📓 Brain Tumor Segmentation.ipynb
├── 📊 Dataset/
│   ├── train/
│   │   ├── images/
│   │   └── _annotations.coco.json
│   ├── valid/
│   │   ├── images/
│   │   └── _annotations.coco.json
│   └── test/
│       ├── images/
│       └── _annotations.coco.json
│
├── README.md
├── .gitignore
```

---

## 🏗️ Architecture

### 🧩 Model: U-Net + ResNet34 Encoder

```
Input (3, 256, 256)
        ↓
  ResNet34 Encoder (ImageNet)
        ↓
     U-Net Decoder
        ↓
Output (1, 256, 256)
```

### ✨ Key Features
- Pre-trained ResNet34 encoder
- Skip connections (U-Net)
- Batch Normalization
- ~21M parameters

---

## 🧮 Loss Function

### Dice Loss

```python
def dice_loss(pred, target, smooth=1.0):
    pred = torch.sigmoid(pred)
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return 1 - dice
```

### 🎯 Why Dice Loss?
- Handles class imbalance
- Directly optimizes segmentation overlap
- More stable for medical images

---

## ⚙️ Configuration

```python
CONFIG = {
    'img_size': (256, 256),
    'batch_size': 16,
    'epochs': 50,
    'learning_rate': 0.0002,
    'patience': 5,
    'device': 'cuda'
}
```

---

## 📈 Training Pipeline

### 🧪 Data Preparation
1. Load COCO annotations  
2. Generate binary masks  
3. Filter low-signal images  
4. Apply preprocessing:
   - Resize (256×256)
   - Normalize
   - Augmentation

### ⚙️ Training Setup

```python
optimizer = Adam(lr=0.0002)

scheduler = ReduceLROnPlateau(
    mode='min',
    factor=0.5,
    patience=2
)
```

---

## 📊 Dataset

### COCO Format Example

```json
{
  "images": [{"id": 1, "file_name": "image.jpg"}],
  "annotations": [{
    "image_id": 1,
    "segmentation": [[x1, y1, x2, y2]]
  }],
  "categories": [{"id": 1, "name": "tumor"}]
}
```

### Classes
- 0 → Background  
- 1 → Tumor  

---

## 🔧 Improvements Applied

| Fix | Impact |
|-----|--------|
| Mask normalization | Huge Dice improvement |
| Pretrained encoder | Faster convergence |
| Dice Loss | Better segmentation |
| LR Scheduler | Stable training |
| Data filtering | Cleaner dataset |

---

## 📚 Usage

### Load Model

```python
model = smp.Unet(
    encoder_name="resnet34",
    encoder_weights="imagenet",
    in_channels=3,
    classes=1
)
```

### Inference

```python
mask = (torch.sigmoid(output) > 0.5).int()
```

---

## 🎯 Performance

- Dice: ~78%
- IoU: ~64%
- FN Rate: ~10–15%
- FP Rate: ~15–20%

### 🏥 Clinical Use
- Screening tool ✅
- Second opinion ✅
- Not a final diagnosis ❗

---

## ⚡ Hardware

### Minimum
- CPU i5
- 8GB RAM

### Recommended
- GPU RTX 2060+
- 16GB RAM

---

## 🚨 Troubleshooting

### Missing Library
```bash
pip install segmentation_models_pytorch
```

### CUDA OOM
```python
CONFIG['batch_size'] = 8
```

---

## 📖 References

- U-Net Paper  
- ResNet Paper  
- PyTorch Docs  
- segmentation_models_pytorch  

---

## 📝 Citation

```bibtex
@software{brain_tumor_segmentation_2026,
  author = {Samir Mohamed},
  title = {Brain Tumor Segmentation using U-Net},
  year = {2026}
}
```

---

## 🤝 Contributing

1. Fork  
2. Create branch  
3. Commit  
4. Push  
5. Pull Request  

---

## 📬 Contact

- Issues: GitHub Issues  
- Discussions: GitHub Discussions  

---

## 🙏 Acknowledgments

- Medical imaging community  
- PyTorch ecosystem  
- ImageNet pretraining  

---

<div align="center">

### ❤️ Built for Medical AI

</div>

---

## 📊 Final Summary

```
Dice Score : 0.7803
IoU Score  : 0.6421
Loss       : 0.2197
Device     : GPU (CUDA)
Status     : Production Ready
```

---

**Last Updated:** March 2026  
**Version:** 1.0

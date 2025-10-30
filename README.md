# PixelRNN: Autoregressive Image Modeling on CIFAR-10

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Implementation and comparative evaluation of three autoregressive generative models for image synthesis: **PixelCNN**, **Row LSTM**, and **Diagonal BiLSTM**, based on the seminal [Pixel Recurrent Neural Networks](https://arxiv.org/abs/1601.06759) paper by van den Oord et al. (2016).

![CIFAR-10 Samples](https://img.shields.io/badge/Dataset-CIFAR--10-orange)

---

## 📋 Table of Contents

- [Overview](#overview)
- [Architecture Implementations](#architecture-implementations)
- [Results](#results)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Methodology](#methodology)
- [Challenges & Solutions](#challenges--solutions)
- [Future Work](#future-work)
- [References](#references)
- [Author](#author)
- [License](#license)

---

## 🎯 Overview

This project implements three autoregressive architectures for pixel-level image generation:

1. **PixelCNN** - Uses masked convolutions for parallel training while maintaining autoregressive dependencies
2. **Row LSTM** - Processes images sequentially row-by-row to capture long-range dependencies
3. **Diagonal BiLSTM** - Processes pixels along diagonals using bidirectional LSTMs for enhanced spatial modeling

All models were trained on the **CIFAR-10 dataset** (32×32 RGB images) using discrete softmax classification over 256 pixel values, evaluated using **bits-per-dimension (BPD)** as the primary metric.

### Key Highlights

- ✅ Faithful reproduction of PixelRNN paper architectures
- ✅ Proper masked convolutions (Type A and Type B)
- ✅ Comprehensive training with validation monitoring
- ✅ Comparative analysis with quantitative metrics
- ✅ Memory-efficient implementation for resource-constrained environments

---

## 🏗️ Architecture Implementations

### 1. PixelCNN

**Architecture:**
- 7×7 masked convolution (Type A) for input layer
- 4 residual blocks with 3×3 masked convolutions (Type B)
- Output layer with 256-way softmax per pixel

**Parameters:** 44,704  
**Key Feature:** Parallel training with masked convolutions maintaining causal dependencies

### 2. Row LSTM

**Architecture:**
- Masked convolution for input projection
- 2 LSTM layers with 32 hidden channels
- Sequential row-by-row processing

**Parameters:** 43,136  
**Key Feature:** Captures long-range intra-row dependencies through recurrent connections

### 3. Diagonal BiLSTM

**Architecture:**
- Masked convolution for input projection
- 1 bidirectional LSTM layer (32 hidden channels)
- Diagonal skewing/unskewing operations

**Parameters:** 32,640  
**Key Feature:** Processes pixels along diagonals for richer spatial context

---

## 📊 Results

### Performance Comparison

| Model | Parameters | Final Train BPD | Final Val BPD | Best Val BPD |
|-------|-----------|-----------------|---------------|--------------|
| **PixelCNN** | 44,704 | 0.0019 | 0.0018 | **0.0018** |
| **Row LSTM** | 43,136 | 0.0019 | 0.0019 | 0.0019 |
| **Diagonal BiLSTM** | 32,640 | 0.0020 | 0.0020 | 0.0020 |

### Key Findings

- **PixelCNN** achieved the best validation BPD (0.0018) with stable training dynamics
- All models demonstrated convergence within 10 epochs
- PixelCNN's convolutional parallelism provides optimal balance of performance and efficiency
- Row LSTM shows competitive performance with sequential modeling
- Diagonal BiLSTM exhibits higher variance but captures complex diagonal relationships

### Training Curves

All models showed stable convergence with proper gradient clipping and learning rate scheduling. Detailed training/validation curves are available in the [full report](PixelRNN_Report.pdf).

---

## 🚀 Installation

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended)
- 8GB+ RAM

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/PixelRNN-CIFAR10-Implementation.git
cd PixelRNN-CIFAR10-Implementation

# Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install torch torchvision numpy matplotlib tqdm
```

### Dependencies

```
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.24.0
matplotlib>=3.7.0
tqdm>=4.65.0
```

---

## 💻 Usage

### Training Models

Open and run the Jupyter notebook:

```bash
jupyter notebook pixelrnn_implementation.ipynb
```

The notebook contains:
1. **Data Loading** - CIFAR-10 dataset preparation
2. **Model Definitions** - All three architectures
3. **Training Loop** - With validation and metric tracking
4. **Evaluation** - BPD computation and visualization
5. **Comparison** - Side-by-side model analysis

### Quick Start

```python
import torch
from torchvision import datasets, transforms

# Load CIFAR-10
transform = transforms.Compose([transforms.ToTensor()])
train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)

# Initialize model (example: PixelCNN)
model = PixelCNN(in_channels=3, hidden_channels=32, num_classes=256)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

# Training loop
for epoch in range(10):
    train_loss = train_epoch(model, train_loader, optimizer)
    val_loss = validate(model, val_loader)
    print(f"Epoch {epoch}: Train BPD={train_loss:.4f}, Val BPD={val_loss:.4f}")
```

### Generating Samples

```python
# Generate new images (autoregressive sampling)
model.eval()
samples = model.generate(num_samples=16, device='cuda')
```

---

## 📁 Project Structure

```
PixelRNN-CIFAR10-Implementation/
│
├── pixelrnn_implementation.ipynb   # Main implementation notebook
├── PixelRNN_Report.pdf             # Detailed technical report
├── README.md                        # This file
├── LICENSE                          # MIT License
│
├── data/                            # CIFAR-10 dataset (auto-downloaded)
│   └── cifar-10-batches-py/
│
└── outputs/                         # Generated samples and plots (created during training)
    ├── training_curves.png
    ├── comparison_plot.png
    └── generated_samples.png
```

---

## 🔬 Methodology

### Training Configuration

- **Optimizer:** Adam (lr=1e-3, weight_decay=1e-4)
- **Scheduler:** ReduceLROnPlateau (factor=0.5, patience=2)
- **Epochs:** 10
- **Batch Size:** 32 (memory-constrained)
- **Gradient Clipping:** Max norm 1.0
- **Loss Function:** Cross-entropy over 256 pixel classes

### Evaluation Metric

**Bits-per-Dimension (BPD):**

```
BPD = NLL / (H × W × C × log(2))
```

Where:
- NLL = Negative Log-Likelihood
- H, W = Image height and width (32)
- C = Number of channels (3 for RGB)

Lower BPD indicates better model performance.

### Masked Convolutions

- **Type A Mask:** Used in first layer, excludes center pixel
- **Type B Mask:** Used in subsequent layers, includes center pixel
- Ensures autoregressive property: pixel prediction depends only on previously generated pixels

---

## 🛠️ Challenges & Solutions

### 1. GPU Memory Constraints
**Challenge:** Limited VRAM on Google Colab  
**Solution:** Reduced batch size to 32, limited hidden dimensions, implemented gradient accumulation, regular CUDA cache clearing

### 2. Numerical Stability
**Challenge:** Gradient explosion in recurrent models  
**Solution:** Gradient clipping (max norm 1.0), careful weight initialization, learning rate scheduling

### 3. Autoregressive Dependencies
**Challenge:** Maintaining causal ordering in parallel architectures  
**Solution:** Proper masked convolution implementation with Type A/B masks

### 4. Training Efficiency
**Challenge:** Slow convergence of recurrent models  
**Solution:** Optimized input preprocessing, efficient masking operations, reduced model complexity

---

## 🔮 Future Work

### Planned Enhancements

1. **Larger Networks** - Scale up hidden dimensions and depth
2. **Extended Training** - Train for 50+ epochs with early stopping
3. **Self-Attention** - Integrate attention mechanisms for long-range dependencies
4. **Conditional Generation** - Class-conditional image synthesis
5. **Advanced Metrics** - Inception Score (IS) and Fréchet Inception Distance (FID)
6. **Sample Quality** - Qualitative evaluation of generated images
7. **Multi-Scale Architecture** - Hierarchical pixel generation
8. **Mixture of Logistics** - Replace softmax with continuous distributions

### Research Directions

- Hybrid architectures combining CNN and LSTM strengths
- Application to higher-resolution datasets (ImageNet, CelebA)
- Real-time generation optimization
- Comparison with modern generative models (VAEs, GANs, Diffusion Models)

---

## 📚 References

1. **van den Oord, A., Kalchbrenner, N., & Kavukcuoglu, K. (2016).** *Pixel Recurrent Neural Networks.* ICML 2016. [arXiv:1601.06759](https://arxiv.org/abs/1601.06759)

2. **Krizhevsky, A., & Hinton, G. (2009).** *Learning Multiple Layers of Features from Tiny Images.* Technical Report, University of Toronto.

3. **van den Oord, A., et al. (2016).** *Conditional Image Generation with PixelCNN Decoders.* NeurIPS 2016. [arXiv:1606.05328](https://arxiv.org/abs/1606.05328)

4. **Salimans, T., et al. (2017).** *PixelCNN++: Improving the PixelCNN with Discretized Logistic Mixture Likelihood and Other Modifications.* ICLR 2017. [arXiv:1701.05517](https://arxiv.org/abs/1701.05517)

---

## 👤 Author

**Huzaifa Nasir**  
 
Department of Computer Science  
National University of Computer and Emerging Sciences (FAST-NUCES), Islamabad, Pakistan

- GitHub: [@yourusername](https://github.com/huzaifanasir95)
- Email: nasirhuzaifa95@Gmail.com

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- Original PixelRNN paper authors for groundbreaking work in autoregressive modeling
- CIFAR-10 dataset creators
- PyTorch community for excellent documentation and tools
- Google Colab for providing free GPU resources

---

## 📈 Citation

If you use this implementation in your research, please cite:

```bibtex
@misc{nasir2024pixelrnn,
  author = {Nasir, Huzaifa},
  title = {PixelRNN Implementations: PixelCNN, Row LSTM, and Diagonal BiLSTM},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/Huzaifanasir95/PixelRNN-CIFAR10-Implementation.git}
}
```

---

## ⭐ Star History

If you find this project helpful, please consider giving it a star! ⭐

---

**Last Updated:** October 2024

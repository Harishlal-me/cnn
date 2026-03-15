# MultiModal TaskGate 🚪

A sophisticated multimodal Deep Learning pipeline designed to perform joint inference across three distinct tasks using both image and text inputs:
1. **Fake News Detection** (Real vs. Fake)
2. **Sentiment Analysis** (Negative, Neutral, Positive)
3. **Harmful Content Detection** (Safe vs. Harmful/Hate-speech)

## 🧠 Architecture Overview
The model uses a gated multimodal fusion architecture:
- **Text Backbone:** `RoBERTa-base` (upgraded for robust contextual embeddings) + MultiScale CNN for character/word-level feature extraction.
- **Image Backbone:** `ResNet50` (layer3 & layer4 fine-tuned).
- **OCR Pipeline:** Integrates `EasyOCR` to automatically extract text tokens from image-only samples (e.g., Memes without associated text).
- **Fusion:** Uses Cross-Modal Gates to dynamically weight visual vs. textual features based on input modality presence.
- **Task Routing:** Dedicated task heads process the fused embeddings, applying weighted cross-entropy (for class imbalances) and Focal Loss (for minority classes).

## 📊 Datasets
Operates on a unified dataset index combining ~52,000 samples from:
- **Fake News:** GossipCop, PolitiFact, BuzzFeed
- **Sentiment & Harmful:** Hateful Memes, Memotion 7K, MVSA-Single, HarM, MMHS150K
- **Splits:** Stratified 80/10/10 (Train/Val/Test).

## 🚀 Training Features
The training pipeline (`run_full_training.py`) is highly resilient and optimized:
- **Curriculum Learning (3 Stages):** Progressively unfreezes model layers (Heads -> Image Encoder -> RoBERTa).
- **Auto-Resume:** Automatically detects previous checkpoints and resumes interrupted sessions seamlessly.
- **Cosine Scheduler:** Linear warmup for 500 steps, followed by a cosine decay.
- **Augmentation:** Train-time image augmentations (Rotation, Color Jitter, Horizontal Flips).
- **Safety Measures:** Gradient clipping, automated NaN batch skipping, protections against PIL Decompression Bombs, and AMP (Automatic Mixed Precision).

## ⚙️ How to Run

1. **Train the Model**
   ```bash
   python training/run_full_training.py
   ```
2. **Evaluate Checkpoints**
   ```bash
   python scripts/evaluate_model.py
   ```

## 🛠️ Requirements
- Python 3.9+
- PyTorch 2.0+ (CUDA 12.1+ recommended)
- Transformers, EasyOCR, Pandas, scikit-learn

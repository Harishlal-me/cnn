# MultiModal TaskGate 🧠🖼️
### Cross-Modal Gated CNN–Transformer Fusion for Multi-Task Misinformation and Hate Detection

![Python](https://img.shields.io/badge/Python-3.10-blue?style=flat-square&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0-orange?style=flat-square&logo=pytorch)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-yellow?style=flat-square)
![EasyOCR](https://img.shields.io/badge/OCR-EasyOCR-green?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)
![Status](https://img.shields.io/badge/Status-In%20Progress-orange?style=flat-square)

---

## 📌 Overview

**MultiModal TaskGate** accepts two input modes — raw text or an image — and simultaneously detects fake news, analyzes sentiment, and identifies harmful content.

The model processes images through two parallel branches: OCR-extracted text and visual CNN features, which are fused via a **cross-modal gating mechanism**. Both text and image inputs share the same downstream pipeline of Multi-Scale CNN + DistilBERT Transformer + Hierarchical Dynamic Gates.

> **Honest framing:** The novelty of this work is architectural and empirical. The contribution stands if the ablation experiments demonstrate that each gating level provides measurable, statistically significant improvement over static fusion baselines.

### Two Input Modes

| Mode | Input | Processing |
|------|-------|-----------|
| ✍️ Text | Raw text | Multi-Scale CNN + DistilBERT + Gates |
| 🖼️ Image | Meme / screenshot / news photo | OCR → text branch + ResNet50 → visual branch → cross-modal gate |

### What It Detects

- 📰 **Fake News Detection**
- 💬 **Sentiment Analysis**
- ⚠️ **Harmful / Hateful Content Detection**

---

## 🎯 Key Contributions

1. **Hierarchical Gating Architecture** — jointly models token-level, task-level, and cross-modal fusion through three learnable sigmoid gates with entropy-based regularization to prevent collapse
2. **Cross-Modal Fusion with OCR Pipeline** — end-to-end pipeline from raw image to multi-task prediction via EasyOCR text extraction + ResNet50 visual features + dynamic cross-modal gate
3. **Curriculum Training Strategy** — three-stage training (text-only → add image → enable cross-modal gate) for stable multimodal convergence on limited compute
4. **Robustness Experiments** — systematic evaluation under OCR noise and missing modality conditions
5. **Comprehensive Interpretability** — token gate heatmaps, task gate analysis, and cross-modal gate visualization with honest error analysis

---

## 🏗️ Architecture

### Quick Summary

```
Image input
  → EasyOCR (text extraction)         → [OCR text]
  → ResNet50 (visual features)         → [1 × 2048] → Linear(2048→512)

Text input (or OCR text)
  → Multi-Scale CNN (k=3,5,7)          → [seq_len × 512]
  → DistilBERT encoder                 → [seq_len × 768] → Linear(768→512)
  → Token-Level Gate                   → per-word CNN/TF blend
  → Task-Level Gate                    → per-task global balance

Both branches
  → Cross-Modal Fusion Gate            → dynamic text vs image weighting
  → Task-Specific Heads                → Fake News / Sentiment / Harmful
```

### Full Pipeline Diagram

```
╔══════════════════════╗     ╔══════════════════════════════╗
║    TEXT MODE  ✍️      ║     ║       IMAGE MODE  🖼️          ║
║  User types text     ║     ║  User uploads image           ║
╚══════════╦═══════════╝     ╚═══════╦══════════════╦════════╝
           ║                         ║              ║
           ║                   [EasyOCR]      [ResNet50]
           ║                   OCR text       Visual features
           ║                         ║        [1×2048]→[1×512]
           ╚═════════════╦═══════════╝              ║
                         ║                          ║
            Tokenization + Embeddings               ║
                         ║                          ║
           ┌─────────────╨──────────┐               ║
           ║                        ║               ║
  [Multi-Scale CNN]         [DistilBERT]            ║
  k=3,5,7 → concat          Self-attention          ║
  [seq_len×384]→[seq_len×512] [seq_len×768]→[seq_len×512]
           ║                        ║               ║
           ╚────────────╦───────────╝               ║
                        ║                           ║
          ┌─────────────╨─────────────┐             ║
          ║    TOKEN-LEVEL GATE 🔑    ║             ║
          ║  g_i=σ(W·[CNN_i;TF_i])   ║             ║
          ║  h_i=g_i×CNN_i+(1-g_i)×TF_i ║          ║
          └─────────────╦─────────────┘             ║
                        ║                           ║
               Mean Pool → [1×512]                  ║
                        ║                           ║
          ┌─────────────╨─────────────┐             ║
          ║    TASK-LEVEL GATE 🔑     ║             ║
          ║  g=σ(W·[h_agg;TF_cls;task])║            ║
          └─────────────╦─────────────┘             ║
                        ║                           ║
                        ╠═══════════════════════════╣
                        ║     (Image Mode only)     ║
          ┌─────────────╨───────────────────────────╨──┐
          ║       CROSS-MODAL FUSION GATE 🔑            ║
          ║  g=σ(W·[text_feat;visual_proj])             ║
          ║  out=g×text+(1-g)×visual                    ║
          ║  (bypassed in Text Mode)                    ║
          └─────────────────────────╦──────────────────┘
                                    ║
                   [Task-Specific Classification Heads]
                   ├── Fake News       (binary)
                   ├── Sentiment       (multi-class)
                   └── Harmful Content (binary)
```

### Why Three Gates?

CNN encoders capture local phrase patterns while Transformers capture long-range dependencies. The relative importance of these representations varies across tasks and inputs — sentiment is keyword-driven; fake news requires discourse-level reasoning. In multimodal inputs, the diagnostic signal may reside primarily in text (a hateful caption) or the image (a violent photograph). Three gating levels allow the model to adapt at word, task, and modality granularity rather than committing to a fixed fusion ratio.

| Gate | Level | Controls |
|------|-------|----------|
| Token-Level | Per word | CNN vs Transformer weight per token |
| Task-Level | Per task | Global CNN/TF balance per task |
| Cross-Modal | Per input | Text vs visual feature weighting |

---

## 📊 Datasets

| Task | Dataset | Used Size | Link |
|------|---------|-----------|------|
| Fake News | Fakeddit | ~50K (multimodal subset) | [GitHub](https://github.com/entitize/Fakeddit) |
| Fake News | NewsCLIPpings | 71K | [GitHub](https://github.com/g-luo/news_clippings) |
| Hateful Memes | Facebook Hateful Memes | 10K | [HuggingFace](https://huggingface.co/datasets/limjiayi/hateful_memes_expanded) |
| Sentiment | MVSA | 4K | [MVSA](http://mcrlab.net/research/mvsa-sentiment-analysis-on-multi-view-social-data/) |
| Harmful Memes | HarM | 3.5K | [GitHub](https://github.com/di-dimitrov/mmf-harmeme) |

> Fakeddit is used as a ~50K multimodal subset (image + text pairs only). Training on the full 1M dataset is not feasible on Colab/Kaggle T4.

### Dataset Splits

Official splits used where available. Otherwise: Train 70% / Val 10% / Test 20% with seed=42.

---

## ⚙️ Installation

```bash
git clone https://github.com/Harishlal-me/MultiModalTaskGate.git
cd MultiModalTaskGate
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

---

## 🚀 Training

Training uses a 3-stage curriculum for stable multimodal convergence:

```bash
# Stage 1 — Text pipeline only
python training/train.py --stage 1 --epochs 5 --batch_size 32 --lr 2e-5

# Stage 2 — Add image encoder
python training/train.py --stage 2 --epochs 5 --batch_size 16 --lr 1e-5 \
  --freeze_distilbert_layers 4 --freeze_resnet_layers 6

# Stage 3 — Enable cross-modal gate
python training/train.py --stage 3 --epochs 5 --batch_size 16 --lr 5e-6 \
  --gate_reg_lambda 0.1 --mixed_precision True
```

### Reproducibility

```yaml
seed: 42          # also run 123 and 2024 for variance reporting
optimizer: AdamW
lr: 2e-5
scheduler: linear_decay_with_warmup
dropout: 0.1
cnn_kernels: [3, 5, 7]
gate_hidden_dim: 256
gate_reg_lambda: 0.1
max_seq_len: 128
image_size: 224x224
ocr_confidence_threshold: 0.6
```

---

## 📈 Results

> Results will be updated after experiments are complete. Three seeds (42, 123, 2024) are used — mean ± std reported.

### Baselines

| Model | Type | Reason Selected |
|-------|------|----------------|
| DistilBERT | Text-only | Strong lightweight text baseline |
| CLIP | Multimodal | Pretrained contrastive vision-language model |
| Late Fusion (concat) | Classical multimodal | Standard static fusion baseline |

### Main Results (Placeholder)

| Model | Fakeddit F1 | Hateful Memes F1 | MVSA F1 | Params |
|-------|------------|-----------------|---------|--------|
| DistilBERT | - | - | - | 66M |
| CLIP | - | - | - | 149M |
| Late Fusion | - | - | - | ~135M |
| **MultiModal TaskGate** | **-** | **-** | **-** | **~95M** |

MultiModal TaskGate is designed to achieve competitive performance with fewer parameters through its hierarchical gating strategy. Empirical results will be reported upon completion.

### Ablation (Placeholder)

> ⚠️ This is the most critical section. If removing any gate shows negligible F1 difference, the core novelty claim weakens. All rows reported with paired t-test (p < 0.05).

| Configuration | Fakeddit F1 | Hateful Memes F1 |
|---------------|------------|-----------------|
| Full model (all 3 gates) | - | - |
| w/o cross-modal gate | - | - |
| w/o token-level gate | - | - |
| w/o task-level gate | - | - |
| No gates (static fusion) | - | - |
| Single CNN kernel k=3 | - | - |
| w/o CNN branch | - | - |
| w/o Transformer branch | - | - |
| w/o visual branch | - | - |
| Single-task training | - | - |

---

## 📂 Project Structure

```
MultiModalTaskGate/
│
├── models/
│   ├── multiscale_cnn.py         # CNN encoder (k=3,5,7)
│   ├── transformer_branch.py     # DistilBERT wrapper
│   ├── token_gate.py             # Token-level gate
│   ├── task_gate.py              # Task-level gate
│   ├── image_encoder.py          # ResNet50 feature extractor
│   ├── cross_modal_gate.py       # Cross-modal fusion gate
│   ├── task_heads.py             # Classification heads
│   └── mm_taskgate.py            # Full model
│
├── input/
│   ├── ocr_extractor.py          # EasyOCR wrapper
│   └── image_preprocessor.py    # Image transforms
│
├── training/
│   ├── train.py                  # Curriculum training loop
│   ├── evaluate.py               # Per-task evaluation
│   └── losses.py                 # Multi-task + gate reg loss
│
├── analysis/
│   ├── token_gate_visualizer.py  # Per-word heatmap
│   ├── task_gate_visualizer.py   # Task gate analysis
│   ├── cross_modal_visualizer.py # Cross-modal gate plots
│   ├── noise_robustness.py       # OCR noise experiments
│   ├── modality_failure.py       # Missing modality tests
│   └── ablation.py               # Ablation runner
│
├── app/
│   └── streamlit_app.py          # Interactive demo
│
├── paper/                        # Full paper details
│   ├── main.tex                  # LaTeX source
│   ├── training_objective.md     # Full loss derivation
│   ├── error_analysis.md         # Detailed failure cases
│   ├── gate_analysis.md          # Gate consistency study
│   └── figures/
│
├── configs/config.yaml
├── requirements.txt
└── README.md
```

---

## ⚠️ Known Limitations

| Issue | Impact | Future Fix |
|-------|--------|------------|
| EasyOCR struggles with stylized/curved meme fonts | ~5% OCR failures | TrOCR or Donut |
| No sarcasm awareness | ~8% misclassification | Sarcasm detection head |
| Satire vs fake news confusion | ~6% errors | Satire-labeled data |
| Visual-only hate symbols (no text) | ~4% errors | Fine-grained visual classifier |

---

## 🚦 Project Status

| Milestone | Status |
|-----------|--------|
| Architecture design | ✅ Complete |
| README + research proposal | ✅ Complete |
| PyTorch implementation | 🔄 In Progress |
| Dataset pipeline | ⏳ Upcoming |
| Training experiments | ⏳ Upcoming |
| Ablation + gate analysis | ⏳ Upcoming |
| Streamlit demo | ⏳ Upcoming |
| Paper draft | ⏳ Upcoming |
| arXiv submission | ⏳ Upcoming |

---

## 🎓 Target Venues

| Venue | Type | Difficulty |
|-------|------|------------|
| arXiv | Preprint | ⭐ |
| IEEE Access | Open Journal (Rolling) | ⭐⭐⭐ |
| IEEE Transactions on Multimedia | Journal | ⭐⭐⭐⭐ |
| EMNLP 2026 | Top Conference | ⭐⭐⭐⭐⭐ |
| COLING 2026 | Top Conference | ⭐⭐⭐⭐ |

---

## 👤 Author

**Harish Lal**
B.Tech CSE (Minor: AI & ML) — SRM Institute of Science and Technology, Chennai
🌐 [Portfolio](https://harishlal-me.vercel.app) · 💻 [GitHub](https://github.com/Harishlal-me) · 📧 meharishlal@gmail.com

---

## 📄 Citation

```bibtex
@article{harishlal2026mmtaskgate,
  title   = {MultiModal TaskGate: Cross-Modal Gated CNN-Transformer Fusion
             for Multi-Task Misinformation and Hate Detection},
  author  = {Harish Lal},
  journal = {arXiv preprint},
  year    = {2026}
}
```

---

## 📜 License

MIT License — see [LICENSE](LICENSE) for details.

---

<p align="center">
  <i>Independent Research Project · SRM Institute of Science and Technology · 2026</i>
</p>

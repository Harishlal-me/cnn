# MultiModal TaskGate — Full Project Document
### Complete Technical Reference: Idea, Architecture, Training, Experiments, and Paper Strategy

> This document contains the complete explanation of the MultiModal TaskGate research project — from the original idea and motivation, through every architectural decision, training strategy, experimental design, and publication plan. It is intended as the single source of truth for the project.

---

## Table of Contents

1. [Project Origin & Motivation](#1-project-origin--motivation)
2. [Problem Statement](#2-problem-statement)
3. [What Makes This Novel](#3-what-makes-this-novel)
4. [Input Modes](#4-input-modes)
5. [Complete Architecture](#5-complete-architecture)
   - 5.1 Multi-Scale CNN Encoder
   - 5.2 Transformer Encoder (DistilBERT)
   - 5.3 Projection Layers
   - 5.4 Token-Level Gate
   - 5.5 Task-Level Gate
   - 5.6 Visual Feature Extractor
   - 5.7 Cross-Modal Fusion Gate
   - 5.8 Task-Specific Heads
6. [Complete Pipeline Step by Step](#6-complete-pipeline-step-by-step)
7. [Training Objective & Loss Functions](#7-training-objective--loss-functions)
8. [Gate Regularization](#8-gate-regularization)
9. [Curriculum Training Strategy](#9-curriculum-training-strategy)
10. [Datasets](#10-datasets)
11. [Baselines](#11-baselines)
12. [Experiments & Evaluation Plan](#12-experiments--evaluation-plan)
13. [Gate Analysis & Interpretability](#13-gate-analysis--interpretability)
14. [Robustness Experiments](#14-robustness-experiments)
15. [Error Analysis](#15-error-analysis)
16. [Known Limitations](#16-known-limitations)
17. [Publication Strategy](#17-publication-strategy)
18. [Development Timeline](#18-development-timeline)
19. [Reproducibility Settings](#19-reproducibility-settings)

---

## 1. Project Origin & Motivation

### Where This Idea Came From

This project grew out of prior work on **CyberGuard** — a BERT-based cyberbullying detection system built using NLP and transformer models. That project was text-only: given a piece of text, classify whether it was cyberbullying.

The limitation of text-only detection became clear quickly. In the real world, misinformation and hateful content rarely exists as plain text alone. It appears as:

- Memes with overlaid text on images
- Screenshots of fake news articles
- Social media posts combining image and caption
- WhatsApp forwards with embedded headlines

A model that only reads text misses the full picture. A model that only looks at images misses the words. The gap between these two is where MultiModal TaskGate lives.

### Why CNN + Transformer?

From the CyberGuard and TaskGate text projects, a clear insight emerged: CNN encoders are strong at capturing **local phrase-level patterns** (keywords, n-grams, sentiment phrases) while Transformer encoders are strong at **long-range contextual understanding** (discourse, contradiction, framing). Neither alone is optimal.

The question became: instead of choosing one or statically combining both, can the model **learn** which one to trust more — per word, per task, and per modality?

That question is the foundation of the entire architecture.

### Why Multi-Task?

Fake news, sentiment, and harmful content are related but distinct signals. A piece of content can be fake AND harmful AND negative in sentiment simultaneously. Training a single model across all three tasks forces the shared encoder to learn representations that generalize across tasks — which is both more efficient and potentially more robust than three separate models.

---

## 2. Problem Statement

**Given:** A piece of content — either raw text, or an image containing embedded text and visual information.

**Goal:** Simultaneously classify the content across three tasks:
- **Fake News Detection** — is this claim or article fabricated or misleading?
- **Sentiment Analysis** — is the sentiment positive, negative, or neutral?
- **Harmful Content Detection** — is this content offensive, hateful, or harmful?

**Challenge:** These tasks have different feature requirements. Sentiment relies heavily on local keyword signals ("terrible", "wonderful"). Fake news detection requires understanding discourse-level patterns and contextual relationships. Harmful content requires both. A static fusion model cannot adapt to this variability.

**Proposed Solution:** A hierarchical gating architecture that dynamically adjusts feature fusion at three levels — per token, per task, and per modality.

---

## 3. What Makes This Novel

### The Core Claim

We propose a **hierarchical gating framework** that jointly models token-level, task-level, and modality-level fusion within a unified multimodal architecture — providing fine-grained, task-aware, and modality-adaptive feature blending at every stage of the pipeline.

### What Exists Already

| Existing Work | What It Does | What It Misses |
|---------------|-------------|----------------|
| TextCNN | CNN for text classification | No Transformer, no gating, no multimodal |
| BERT / DistilBERT | Transformer for text | No CNN, no gating, no multimodal |
| MT-DNN | Multi-task Transformer | No CNN, no gating, no multimodal |
| VisualBERT / LXMERT | Multimodal fusion | Static fusion, no hierarchical gating |
| CLIP | Contrastive vision-language alignment | Not a classifier, no task-specific gating |
| Simple concat fusion | CNN + image features concatenated | Static ratio, no per-token or per-task adaptation |

### What This Work Adds

- **Token-level gating** — each word gets its own CNN/Transformer blend weight
- **Task-level gating** — each task gets its own global feature balance
- **Cross-modal gating** — each input gets its own text/visual blend weight
- **All three combined** in one unified model with shared encoder and task-specific heads
- **Entropy-based regularization** to prevent any gate from collapsing to one branch
- **Curriculum training** specifically designed for stable multimodal convergence

### Honest Assessment of Novelty

The individual components (CNN, Transformer, gating, multi-task) are not individually new. The novelty is in the **combination and the systematic three-level hierarchy** applied to multimodal text detection. This is architectural and empirical novelty, not conceptual novelty.

**The paper will stand or fall on the ablation study.** If removing any single gate produces negligible F1 improvement, the reviewer will conclude the gates are unnecessary. The experimental design must be rigorous.

---

## 4. Input Modes

The model accepts two distinct input modes, both feeding into the same downstream pipeline.

### Text Mode

User provides raw text directly — a tweet, article paragraph, comment, or any string.

```
Input:  "This article claims vaccines cause autism and are dangerous."
          ↓
Tokenization (DistilBERT WordPiece)
          ↓
Multi-Scale CNN + DistilBERT + Token Gate + Task Gate
          ↓
Cross-modal gate bypassed → text features used directly
          ↓
Task-specific classification heads
```

### Image Mode

User uploads an image. The model processes it through two parallel branches.

```
Input: [meme / screenshot / news photo / social post]
          ↓
          ├─────────────────────────────────┐
          │                                 │
    [EasyOCR]                        [ResNet50]
    Extract embedded text            Extract visual features
    "vaccines cause autism"          [1 × 2048]
          │                                 │
    [Text Pipeline]                  Linear(2048 → 512)
    CNN + Transformer                [1 × 512]
    + Token Gate                             │
    + Task Gate                              │
    [1 × 512]                               │
          │                                 │
          └──────────────┬──────────────────┘
                         │
              [Cross-Modal Fusion Gate]
                         │
              Task-specific classification heads
```

**Key design decision:** In Image Mode, the cross-modal gate is active. In Text Mode, the visual branch is bypassed and the gate defaults to text features only. This makes the model backward-compatible with text-only inputs.

---

## 5. Complete Architecture

### 5.1 Multi-Scale CNN Encoder

**Purpose:** Capture local phrase-level patterns at multiple granularities simultaneously.

**Design:**
```
Input embeddings: [seq_len × embed_dim]

Conv1D (kernel=3, filters=128) → ReLU → Dropout(0.1) → [seq_len × 128]
Conv1D (kernel=5, filters=128) → ReLU → Dropout(0.1) → [seq_len × 128]
Conv1D (kernel=7, filters=128) → ReLU → Dropout(0.1) → [seq_len × 128]

Concatenate along feature dim → [seq_len × 384]
Linear(384 → 512) → [seq_len × 512]   ← projection for gate compatibility
```

**Why three kernels:**

| Kernel | Phrase Length | Example |
|--------|-------------|---------|
| 3 | Short phrases | *"cause autism"*, *"very good"* |
| 5 | Medium phrases | *"not worth buying at all"* |
| 7 | Longer expressions | *"extremely disappointing experience overall"* |

Sentiment keywords tend to be short (kernel=3). Fake news phrasing tends to be medium (kernel=5). Conspiracy language tends to be longer (kernel=7). Having all three lets the model choose.

---

### 5.2 Transformer Encoder (DistilBERT)

**Purpose:** Capture long-range contextual relationships between words.

**Why DistilBERT instead of BERT:**
- Retains ~97% of BERT's performance on downstream tasks
- 40% fewer parameters (66M vs 110M)
- 60% faster inference
- Well-suited for multi-task training on limited compute (Colab/Kaggle T4)

**Design:**
```
Input: tokenized text [seq_len × embed_dim]
DistilBERT 6-layer encoder with self-attention
Output: all hidden states [seq_len × 768]
        + [CLS] token [1 × 768] for task-level gate
```

**What DistilBERT learns that CNN misses:**

```
"The vaccine causes autism claim is false"

Relationships:
  claim   ↔  false      (fact-checking contradiction)
  vaccine ↔  autism     (conspiracy association)
  causes  ↔  claim      (hedging language)
```

---

### 5.3 Projection Layers

**Problem:** CNN outputs `[seq_len × 512]` and DistilBERT outputs `[seq_len × 768]`. These dimensions cannot be directly combined in the gating formula.

**Solution:** Both are projected to a common dimension of 512 before any gating operation.

```
CNN output:         [seq_len × 384] → Linear(384→512) → [seq_len × 512]
Transformer output: [seq_len × 768] → Linear(768→512) → [seq_len × 512]
Visual features:    [1 × 2048]      → Linear(2048→512) → [1 × 512]
```

All three modalities share `dim=512` as the common feature space throughout the gating pipeline.

---

### 5.4 Token-Level Gate

**Purpose:** Every individual word gets its own CNN/Transformer blend weight. This is the finest-grained level of the hierarchical gate.

**Formula:**
```
CNN projected:  cnn_i  ∈ R^512    (for token i)
TF projected:   tf_i   ∈ R^512    (for token i)
Task embedding: task_e ∈ R^64     (learned per task)

g_i = σ(W · [cnn_i ; tf_i ; task_e] + b)    W ∈ R^(1 × (512+512+64))
h_i = g_i × cnn_i + (1 − g_i) × tf_i        h_i ∈ R^512
```

**What this looks like in practice:**

```
Sentence: "This article claims vaccines cause autism"

Token      g_i    Interpretation
──────────────────────────────────────────────────
This       0.20   Transformer-heavy — pure discourse word
article    0.30   Context-dependent
claims     0.50   Equal blend — hedging/framing word
vaccines   0.40   Context needed for meaning
cause      0.50   Relational — needs both
autism     0.80   Strong local signal — CNN heavy
```

The model learns these weights entirely from data — they are never manually set.

**Output:** `[seq_len × 512]` — per-token fused representations.

---

### 5.5 Task-Level Gate

**Purpose:** After token-level fusion, a single sentence-level representation is computed by mean pooling. A second gate then applies a global task-conditioned blend using the [CLS] token from DistilBERT.

**Formula:**
```
h_agg  = mean(h_i for i in 1..seq_len)     ∈ R^512
TF_cls = [CLS] token from DistilBERT        ∈ R^768 → Linear(768→512) → R^512
task_e = task embedding                      ∈ R^64

g_task = σ(W · [h_agg ; TF_cls ; task_e])
final  = g_task × h_agg + (1 − g_task) × TF_cls
```

**Expected behavior per task:**
```
Sentiment task   → g_task → 1   (CNN-heavy — keyword-driven)
Fake news task   → g_task → 0   (TF-heavy  — context-driven)
Harmful content  → g_task ≈ 0.5 (equal blend — needs both)
```

This is the second level of the hierarchy. Note that the task embedding is shared with the token-level gate — so the same task identity signal flows through both gating levels.

---

### 5.6 Visual Feature Extractor

**Purpose:** Extract a visual feature vector from the input image for Image Mode.

**Design:**
```
Input: image [3 × 224 × 224]
ResNet50 (pretrained on ImageNet, early layers frozen during Stage 2)
Global Average Pooling → [1 × 2048]
Linear(2048 → 512) → [1 × 512]    ← projected to common dim
```

**What ResNet captures:**
- Image layout and format (meme template, news article layout)
- Objects, faces, logos present in the image
- Visual style and composition
- Image-text visual inconsistencies (a smiling face paired with angry text)

**Why not CLIP features:** CLIP is used as a standalone baseline for comparison. Using CLIP features inside TaskGate's own visual encoder would make the comparison unfair — CLIP would effectively be present in both the baseline and our model.

---

### 5.7 Cross-Modal Fusion Gate

**Purpose:** Dynamically balance text and visual features per input. This is the third and highest level of the hierarchy — it operates at the modality level.

**Formula:**
```
text_feat:   [1 × 512]  (output of task-level gate)
visual_proj: [1 × 512]  (ResNet50 → Linear projection)

g_modal = σ(W · [text_feat ; visual_proj])
final   = g_modal × text_feat + (1 − g_modal) × visual_proj
```

**Expected behavior per input type:**
```
Meme: hateful text + neutral background
  g_modal → 0.8+   (text carries the signal)

News photo: misleading caption + doctored image
  g_modal ≈ 0.5    (both modalities matter)

Image: hate symbol + no text (OCR returns empty)
  g_modal → 0.0    (model must rely on visual — degraded case)
```

**In Text Mode:** The visual branch is not computed. `g_modal = 1.0` effectively — the cross-modal gate is bypassed and `final = text_feat`.

---

### 5.8 Task-Specific Classification Heads

**Purpose:** Three separate linear classifiers, one per task, operating on the shared fused representation.

```
shared_repr: [1 × 512]  (output of cross-modal gate or task-level gate)

Head A — Fake News:
  Linear(512 → 2) → Softmax → [P(real), P(fake)]

Head B — Sentiment:
  Linear(512 → 3) → Softmax → [P(pos), P(neg), P(neu)]

Head C — Harmful Content:
  Linear(512 → 2) → Softmax → [P(safe), P(harmful)]
```

The shared encoder and gating modules receive gradient signals from all three heads. Only heads corresponding to available labels contribute to the loss for any given sample (see Section 7).

---

## 6. Complete Pipeline Step by Step

### Text Mode — Full Flow

```
Step 1: Raw text input
        "Vaccines are a conspiracy created by pharma companies"

Step 2: Tokenization (DistilBERT WordPiece, max_len=128)
        ["vaccines", "are", "a", "conspiracy", "created", "by", "pharma", "companies"]

Step 3: Embeddings → [seq_len × 768]

Step 4: Multi-Scale CNN
        3 parallel Conv1D (k=3,5,7) + ReLU + Dropout → concat → Linear
        Output: [seq_len × 512]

Step 5: DistilBERT encoder
        Self-attention over full sequence
        Output: [seq_len × 512] (projected) + [CLS] [1 × 512]

Step 6: Token-Level Gate
        Per-token sigmoid gate using CNN + TF + task embedding
        Output: [seq_len × 512]

Step 7: Mean pooling → [1 × 512]

Step 8: Task-Level Gate
        Global gate using pooled rep + [CLS] + task embedding
        Output: [1 × 512]

Step 9: Cross-modal gate bypassed → text features passed directly

Step 10: Task heads
         Fake News → 0.82 (Likely Fake)
         Sentiment → Negative
         Harmful   → 0.71 (Likely Harmful)
```

### Image Mode — Full Flow

```
Step 1: Image input [3 × 224 × 224]

Step 2: Parallel processing:
        Branch A — EasyOCR → extracted text string
        Branch B — ResNet50 → [1 × 2048] → Linear → [1 × 512]

Step 3: OCR text goes through Steps 2–8 of Text Mode
        Output: text_feat [1 × 512]

Step 4: Cross-Modal Fusion Gate
        g_modal = σ(W · [text_feat ; visual_proj])
        final = g_modal × text_feat + (1−g_modal) × visual_proj
        Output: [1 × 512]

Step 5: Task heads → predictions
```

---

## 7. Training Objective & Loss Functions

### Full Loss Formula

```
L_total = m₁·λ₁·L_fake_news
        + m₂·λ₂·L_sentiment
        + m₃·λ₃·L_harmful
        + λg·L_gate_total

Where:
  m_t = 1 if task label available for this sample, else 0
  λ₁  = 1.0   (fake news)
  λ₂  = 0.8   (sentiment)
  λ₃  = 1.2   (harmful — upweighted due to class imbalance)
  λg  = 0.1   (gate regularization)
```

### Per-Task Loss Functions

| Task | Loss | Reason |
|------|------|--------|
| Fake News Detection | Weighted Binary Cross-Entropy | Binary task, class imbalance |
| Sentiment Analysis | Cross-Entropy | Multi-class (3 classes) |
| Harmful Content | Focal Loss (γ=2) | Severe minority class in hate datasets |

### Multi-Task Label Masking

Each dataset provides labels for only its own task. Training is done across all datasets simultaneously with task masking:

```python
L_total = sum(m_t * lambda_t * L_t  for t in [fake_news, sentiment, harmful])

# Fakeddit sample:   m_fake=1, m_sentiment=0, m_harmful=0
# MVSA sample:       m_fake=0, m_sentiment=1, m_harmful=0
# HarM sample:       m_fake=0, m_sentiment=0, m_harmful=1
```

Backpropagation flows through the shared encoder from all active task heads. The shared encoder benefits from gradients across all tasks simultaneously — this is the intended multi-task learning benefit.

### Dataset-Task Mapping

| Dataset | Fake News | Sentiment | Harmful |
|---------|-----------|-----------|---------|
| Fakeddit | ✅ | ❌ masked | ❌ masked |
| NewsCLIPpings | ✅ | ❌ masked | ❌ masked |
| Hateful Memes | ❌ masked | ❌ masked | ✅ |
| HarM | ❌ masked | ❌ masked | ✅ |
| MVSA | ❌ masked | ✅ | ❌ masked |

---

## 8. Gate Regularization

### Problem

Without regularization, sigmoid gates can collapse — learning to always output g≈0 or g≈1, which means the model simply ignores one branch entirely. This defeats the purpose of dynamic gating.

### Solution — Entropy-Based Regularization

Binary entropy is maximized at g=0.5 and goes to 0 at g=0 or g=1. Maximizing entropy during training discourages collapse and encourages the gate to use both branches meaningfully.

```
H(g) = − g·log(g) − (1−g)·log(1−g)

Applied to all three gating layers:
  L_gate_total = mean(H(g_i) for all token gates)
               + H(g_task)
               + H(g_modal)

Final loss:
  L_total = L_task + 0.1 × L_gate_total
```

### Why Entropy Over g(1−g)

The simpler formulation `L = mean(g × (1−g))` has weak gradients near 0 and 1 — the model can still collapse because the penalty decreases as the gate saturates. The entropy formulation maintains gradient signal even at extreme gate values, making it a stronger regularizer. Both are compared in the ablation.

### λ Ablation

| λ | Effect |
|---|--------|
| 0.0 | Gates may collapse to one branch |
| 0.05 | Moderate regularization |
| **0.1** | **Recommended — gates remain diverse** |
| 0.5 | Gates over-constrained near 0.5, loses task adaptivity |

---

## 9. Curriculum Training Strategy

### Why Curriculum Training?

Training image encoder + text encoder + three gates simultaneously from scratch is unstable. The image encoder gradients can dominate early training and corrupt the text representations before the gates learn to control the fusion. This is a known failure mode in multimodal training.

### Three-Stage Protocol

**Stage 1 — Text Pipeline Only (Weeks 9–10)**
```
Active:   Multi-Scale CNN + DistilBERT + Token Gate + Task Gate
Frozen:   ResNet50, Cross-Modal Gate (disabled)
Goal:     Learn stable text representations and gating behavior
Config:   epochs=5, batch=32, lr=2e-5
```

**Stage 2 — Add Image Encoder (Week 10–11)**
```
Active:   Full text pipeline + ResNet50
Frozen:   First 6 ResNet50 blocks, first 4 DistilBERT layers
          Cross-Modal Gate still disabled
Goal:     Learn visual features without disrupting text pipeline
Config:   epochs=5, batch=16, lr=1e-5
```

**Stage 3 — Enable Cross-Modal Gate (Week 11–12)**
```
Active:   Everything, including Cross-Modal Gate
Frozen:   Nothing
Goal:     Learn cross-modal fusion on top of stable unimodal representations
Config:   epochs=5, batch=16, lr=5e-6, gate_reg_lambda=0.1
```

### Why This Is a Strong Contribution

Many multimodal papers fail to converge because they train everything at once. This three-stage protocol is practical, reproducible, and directly addresses a known weakness of joint multimodal training. It is a meaningful contribution **independent of the final accuracy results** — even if the accuracy improvement is modest, the training strategy itself is publishable as a practical finding.

---

## 10. Datasets

### Selected Datasets

| Task | Dataset | Used Size | Modality | Link |
|------|---------|-----------|----------|------|
| Fake News | Fakeddit | ~50K (multimodal subset) | Image + Text | [GitHub](https://github.com/entitize/Fakeddit) |
| Fake News | NewsCLIPpings | 71K | Image + Text | [GitHub](https://github.com/g-luo/news_clippings) |
| Hateful Memes | Facebook Hateful Memes | 10K | Image + Text | [HuggingFace](https://huggingface.co/datasets/limjiayi/hateful_memes_expanded) |
| Sentiment | MVSA | 4K | Image + Text | [MVSA](http://mcrlab.net/research/mvsa-sentiment-analysis-on-multi-view-social-data/) |
| Harmful Memes | HarM | 3.5K | Image + Text | [GitHub](https://github.com/di-dimitrov/mmf-harmeme) |

### Important Notes

**Fakeddit subset:** The full Fakeddit dataset contains 1M+ samples but most are text-only or image-only. Only the multimodal subset (~50K where both image and text exist) is used. Training on the full 1M with image features on Colab/Kaggle T4 is computationally infeasible.

**Class imbalance:** Hateful Memes has ~35% hate class. HarM is similarly imbalanced. This is handled via Focal Loss for harmful content tasks.

### Preprocessing

```
Text:
  Tokenizer          = DistilBERT WordPiece
  Max sequence len   = 128 tokens
  Truncation         = right-side
  Padding            = max length

Image:
  Resolution         = 224 × 224
  Normalization      = ImageNet mean/std ([0.485,0.456,0.406], [0.229,0.224,0.225])
  Augmentation       = random horizontal flip + color jitter (train only)

OCR:
  Engine             = EasyOCR
  Confidence thresh  = 0.6 (tokens below this filtered out)
  Max extracted len  = 128 tokens
  Fallback           = if OCR yields < 3 tokens, visual features used only
```

### Dataset Splits

Official splits used where available. Otherwise 70/10/20 with seed=42.

| Dataset | Split Source |
|---------|-------------|
| Fakeddit | Official |
| Hateful Memes | Official |
| NewsCLIPpings | Official |
| MVSA | Random 70/10/20, seed=42 |
| HarM | Random 70/10/20, seed=42 |

---

## 11. Baselines

Three focused baselines are selected to keep experiments clean:

| Baseline | Type | Justification |
|----------|------|---------------|
| DistilBERT | Text-only | Strong lightweight text baseline; isolates text-only performance |
| CLIP | Multimodal | Pretrained contrastive vision-language model; strong multimodal reference |
| Late Fusion (concat) | Classical multimodal | Static concatenation baseline; directly shows value of dynamic gating |

**Why not more baselines:** More baselines make experiments noisier and harder to reproduce. Three strong, well-justified baselines are more credible than seven weak ones. VisualBERT and LXMERT are noted as related work in the paper but not re-implemented.

**Important:** CLIP features are NOT used inside TaskGate's visual encoder. CLIP is a standalone baseline only. Mixing CLIP features into TaskGate while comparing against CLIP would make the comparison unfair.

---

## 12. Experiments & Evaluation Plan

### Primary Evaluation Metrics

For all tasks and all models:
- **F1 Score** (primary metric)
- **Precision**
- **Recall**
- **ROC-AUC**

F1 is prioritized because all datasets have class imbalance. Accuracy alone is misleading.

### Seed Variance

All experiments run with 3 seeds: **42, 123, 2024**. Mean ± std reported in all result tables.

### Statistical Significance

All comparisons between models reported with **paired t-test (p < 0.05)**. This confirms improvements are not due to random variance.

### Main Results Table Structure

| Model | Fakeddit F1 ± std | Hateful Memes F1 ± std | MVSA F1 ± std | Params |
|-------|-------------------|------------------------|---------------|--------|
| DistilBERT | - | - | - | 66M |
| CLIP | - | - | - | 149M |
| Late Fusion | - | - | - | ~135M |
| **TaskGate** | **-** | **-** | **-** | **~95M** |

### Ablation Table Structure

| Configuration | Fakeddit F1 | Hateful Memes F1 | MVSA F1 |
|---------------|------------|-----------------|---------|
| Full model | - | - | - |
| w/o cross-modal gate | - | - | - |
| w/o token-level gate | - | - | - |
| w/o task-level gate | - | - | - |
| No gates (static fusion) | - | - | - |
| Single kernel k=3 only | - | - | - |
| Two kernels k=3,5 only | - | - | - |
| w/o CNN branch | - | - | - |
| w/o Transformer branch | - | - | - |
| w/o visual branch | - | - | - |
| Single-task training | - | - | - |
| Entropy reg vs g(1−g) reg | - | - | - |
| λ=0.0 vs λ=0.1 vs λ=0.5 | - | - | - |

---

## 13. Gate Analysis & Interpretability

This section is critical — it is the paper's most distinctive contribution. Every gate claim must be backed by visualization and quantitative analysis.

### Token Gate Analysis

**Experiment:** Run the trained model on 100 test sentences per task. Record g_i values for every token. Compute mean CNN weight vs Transformer weight per token position and per POS tag.

**Expected finding:**
```
Sentiment-bearing adjectives (terrible, amazing)  → g_i > 0.7  (CNN-heavy)
Discourse connectors (however, although)           → g_i < 0.3  (TF-heavy)
Named entities (vaccine, government)               → g_i ≈ 0.5  (both)
```

**Paper figure:** Heatmap over a sample sentence with g_i values shown per token as color intensity.

### Task Gate Analysis

**Experiment:** For the same input sentence, run inference with task_id set to each of the three tasks. Record g_task for each.

**Expected finding:**
```
Same sentence, task=sentiment  → g_task ≈ 0.7+ (CNN-heavy)
Same sentence, task=fake_news  → g_task ≈ 0.3  (TF-heavy)
Same sentence, task=harmful    → g_task ≈ 0.5  (equal)
```

**Why this matters:** It proves the task embedding is doing real work — same input, different gate values purely because of task identity.

### Cross-Modal Gate Analysis

**Experiment:** For image mode inputs, record g_modal and compare:
- Inputs where OCR yields rich text vs sparse text
- Inputs where image contains clear semantic content (faces, symbols) vs abstract background

**Expected finding:**
```
Meme with clear hateful caption + neutral background  → g_modal > 0.7
News photo with misleading image + ambiguous caption  → g_modal ≈ 0.5
Abstract image + no OCR text                          → g_modal < 0.2
```

### Gate Consistency Analysis

| Experiment | Compares | Expected Result |
|------------|----------|-----------------|
| Same sentence, 3 tasks | g_task per task | Significant variance across tasks |
| Short vs long sentences | g_i distribution | Long sentences → more TF-heavy tokens |
| Clean OCR vs noisy OCR | g_modal shift | Noisy OCR → g_modal decreases (less trust in text) |

---

## 14. Robustness Experiments

### OCR Noise Robustness

**Motivation:** Real-world meme text is often stylized, curved, or partially readable. The model should degrade gracefully.

**Noise generation:**
```
Original: "vaccines cause autism"
Level 1:  "vacc!nes cause autism"       (mild character substitution)
Level 2:  "v@cc1nes c@use aut!sm"       (moderate corruption)
Level 3:  "v@cc1n3s c@u53 @ut!5m"       (severe corruption)
```

**Expected result:** Token-level gate should down-weight corrupted tokens (low confidence signals from CNN on nonsense text) and rely more on Transformer context and visual features.

**Results table:**

| Model | Clean F1 | Noise L1 | Noise L2 | Noise L3 | Avg Drop |
|-------|---------|---------|---------|---------|----------|
| DistilBERT | - | - | - | - | - |
| Late Fusion | - | - | - | - | - |
| **TaskGate** | **-** | **-** | **-** | **-** | **-** |

### Modality Failure Experiments

**Motivation:** Real social media inputs are messy — images may not download, OCR may fail completely.

| Scenario | Image | Text | Expected Behavior |
|----------|-------|------|-------------------|
| Normal | ✅ clean | ✅ clean | Best performance |
| Missing image | ❌ | ✅ | Cross-modal gate adapts to text-only |
| OCR failed | ✅ | ❌ empty | Gate relies entirely on visual |
| Both degraded | ✅ noisy | ✅ noisy | Graceful degradation |

**Hypothesis:** The cross-modal gate should dynamically adapt rather than failing catastrophically. Static concat models will degrade more sharply.

---

## 15. Error Analysis

Honest analysis of where the model fails. This section is included in the paper to build credibility.

### Case 1 — Sarcasm
```
Image:       Smiling person holding vaccine card
Caption:     "Great job vaccines 🙄"
Prediction:  Not harmful  (confidence: 0.31)
Truth:       Harmful (sarcasm)
Reason:      "Great job" has positive local CNN signal.
             Eye-roll emoji underweighted.
             Sarcasm requires pragmatic world knowledge the model lacks.
```

### Case 2 — Stylized OCR Failure
```
Image:       Meme with decorative font
OCR output:  "v@cc!n3s k!ll p30ple"  (confidence 0.38 — below threshold, filtered)
Prediction:  Uncertain (confidence: 0.48)
Truth:       Harmful
Reason:      OCR confidence below 0.6, tokens filtered.
             Visual features alone insufficient for abstract text meaning.
```

### Case 3 — Satire vs Fake News
```
Image:       Screenshot of The Onion article
Caption:     "CDC Confirms Vaccines Contain 5G Microchips"
Prediction:  Fake news  (confidence: 0.79)
Truth:       Satire (not fake news per se)
Reason:      Satire sites mimic fake news phrasing exactly.
             No satire-awareness signal in training data.
```

### Case 4 — Visual-Only Hate
```
Image:       Hate symbol/logo, no text
OCR output:  [empty — 0 tokens extracted]
Prediction:  Not harmful  (confidence: 0.22)
Truth:       Harmful
Reason:      Gate defaults to text when OCR empty.
             Abstract symbol recognition requires fine-grained visual features
             beyond what ResNet50's global pooling provides.
```

### Summary Table

| Failure Type | Estimated Frequency | Root Cause | Future Fix |
|-------------|--------------------|-----------|-----------| 
| Sarcasm | ~8% of errors | No pragmatic knowledge | Sarcasm detection head |
| Stylized OCR | ~5% of errors | EasyOCR limitation | TrOCR or Donut |
| Satire confusion | ~6% of errors | Training data gap | Satire-labeled data |
| Visual-only hate | ~4% of errors | ResNet global pooling | Fine-grained visual classifier |
| Emoji-heavy text | ~3% of errors | WordPiece splits emojis | Emoji-aware tokenization |

---

## 16. Known Limitations

### Technical Limitations

**OCR Bottleneck:** EasyOCR is the weakest link in the pipeline. It struggles with meme fonts, curved text, very small text, and emoji-embedded text. TrOCR (Microsoft) or Donut (OCR-free document understanding) would be stronger replacements but are more complex to integrate.

**Sarcasm:** The model has no mechanism for detecting sarcasm or irony. These require pragmatic knowledge and context beyond token-level patterns.

**English Only:** The model is trained entirely on English datasets. Code-switched content (Hinglish, Tanglish, Spanglish) or non-English memes are not handled.

**Compute:** Full experiments require GPU access. Colab Pro or Kaggle T4 (free) is sufficient for the dataset sizes used, but training time will be significant.

### Research Limitations

**Novelty is incremental:** The individual components are not individually new. The paper's strength depends entirely on rigorous ablation and gate analysis.

**Dataset distribution shift:** The five datasets come from very different distributions (Reddit, Facebook, Twitter, news). The model's performance may vary significantly across these distributions.

**No real-time deployment:** The pipeline (OCR + ResNet + DistilBERT + gates) is not optimized for real-time use. Inference latency is estimated in the ~80–120ms range but not yet measured.

---

## 17. Publication Strategy

### Two-Paper Strategy

**Paper 1 — Text-Only TaskGate (Faster, lower risk)**
Build Phases 1+2 (Multi-Scale CNN + Token Gate + Task Gate, text only). Submit this first while building Phase 3.
- Title: *"TaskGate: Task-Adaptive Multi-Scale CNN–Transformer with Hierarchical Dynamic Gating for Multi-Domain Text Detection"*
- Target: IEEE Access (rolling) or EMNLP Workshop 2026
- Timeline: Submit ~Week 11

**Paper 2 — MultiModal TaskGate (Main paper)**
Full multimodal system with cross-modal gate.
- Title: *"MultiModal TaskGate: Cross-Modal Gated CNN–Transformer Fusion for Multi-Task Misinformation and Hate Detection"*
- Target: EMNLP 2026 / COLING 2026 / IEEE Transactions on Multimedia
- Timeline: Submit ~Week 20

### Target Venues

| Venue | Type | Deadline | Difficulty |
|-------|------|----------|------------|
| arXiv | Preprint | Anytime | ⭐ |
| IEEE Access | Open Journal | Rolling | ⭐⭐⭐ |
| IEEE Transactions on Multimedia | Journal | Rolling | ⭐⭐⭐⭐ |
| EMNLP 2026 | Top Conference | ~May 2026 | ⭐⭐⭐⭐⭐ |
| COLING 2026 | Top Conference | ~March 2026 | ⭐⭐⭐⭐ |

### Recommended Submission Path

```
Week 11 → arXiv preprint (text-only TaskGate)
Week 11 → IEEE Access submission (text-only)
Week 20 → arXiv preprint (multimodal)
Week 20 → EMNLP 2026 submission (multimodal)
```

---

## 18. Development Timeline

| Week | Phase | Milestone |
|------|-------|-----------|
| 1–2 | Planning | Literature review + related work |
| 3 | Data | EasyOCR integration + image preprocessing pipeline |
| 4 | Model | Multi-Scale CNN encoder (k=3,5,7) |
| 5 | Model | DistilBERT encoder + projection layers |
| 6 | Model | Token-level gate + task-level gate |
| 7 | Model | ResNet50 image encoder |
| 8 | Model | Cross-modal fusion gate |
| 9 | Training | Curriculum Stage 1 — text pipeline training |
| 10 | Training | Curriculum Stage 2 — add image encoder |
| 11 | Training | Curriculum Stage 3 — cross-modal gate |
| 11 | Paper | **Paper 1 draft + arXiv + IEEE Access submission** |
| 12 | Experiments | Baseline experiments + main results |
| 13 | Experiments | Ablation study (all rows, 3 seeds) |
| 14 | Experiments | Gate analysis + all 3 visualizations |
| 15 | Experiments | Noise robustness + modality failure |
| 16 | Experiments | Error analysis |
| 17 | Demo | Streamlit demo app (text + image modes) |
| 18 | Paper | Paper 2 full draft |
| 19 | Paper | Revisions + proofreading |
| 20 | Paper | **arXiv + EMNLP 2026 submission** |

---

## 19. Reproducibility Settings

All experiments use the following fixed configuration:

```yaml
# configs/config.yaml

# Reproducibility
seed: 42              # Also run 123 and 2024 for variance
seeds_all: [42, 123, 2024]

# Model
cnn_kernels: [3, 5, 7]
cnn_filters: 128
projection_dim: 512
gate_hidden_dim: 256
task_embedding_dim: 64
dropout: 0.1
image_encoder: resnet50
transformer: distilbert-base-uncased

# Training
optimizer: AdamW
weight_decay: 0.01
max_grad_norm: 1.0
warmup_steps: 500
scheduler: linear_decay_with_warmup
mixed_precision: true
gradient_accumulation_steps: 4

# Stage-specific
stage1_lr: 2e-5
stage1_epochs: 5
stage1_batch: 32

stage2_lr: 1e-5
stage2_epochs: 5
stage2_batch: 16
freeze_distilbert_layers: 4
freeze_resnet_layers: 6

stage3_lr: 5e-6
stage3_epochs: 5
stage3_batch: 16
gate_reg_lambda: 0.1

# Data
max_seq_len: 128
image_size: 224
ocr_confidence_threshold: 0.6
ocr_min_tokens: 3

# Loss weights
lambda_fake_news: 1.0
lambda_sentiment: 0.8
lambda_harmful: 1.2
focal_loss_gamma: 2.0
```

---

## Project Status

| Component | Status |
|-----------|--------|
| Research idea + problem statement | ✅ Complete |
| Architecture design | ✅ Complete |
| README + project documentation | ✅ Complete |
| PyTorch implementation | 🔄 In Progress |
| Dataset pipeline | ⏳ Upcoming |
| Curriculum training | ⏳ Upcoming |
| Baseline experiments | ⏳ Upcoming |
| Ablation study | ⏳ Upcoming |
| Gate analysis + visualizations | ⏳ Upcoming |
| Noise + modality failure experiments | ⏳ Upcoming |
| Streamlit demo | ⏳ Upcoming |
| Paper 1 draft | ⏳ Upcoming |
| Paper 2 draft | ⏳ Upcoming |

---

## Author

**Harish Lal**
B.Tech CSE (Minor: AI & ML) — SRM Institute of Science and Technology, Chennai
🌐 [Portfolio](https://harishlal-me.vercel.app) · 💻 [GitHub](https://github.com/Harishlal-me) · 📧 meharishlal@gmail.com

---

*Independent Research Project · SRM Institute of Science and Technology · 2026*

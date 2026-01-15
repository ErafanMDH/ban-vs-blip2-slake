# MedVQA Comparison: BAN vs. BLIP-2 (BioGPT)

This project performs a side-by-side comparison of **Discriminative** vs. **Generative** Artificial Intelligence for Medical Visual Question Answering (MedVQA).

We utilize the **SLAKE dataset** (English subset) to evaluate how traditional attention networks compare to modern Large Language Models (LLMs) in a medical context.

## 🏗️ Methodological Architecture

### 1. The Discriminative Model: BAN (Bilinear Attention Network)
* **Vision Encoder:** BiomedCLIP (Microsoft) - *Last transformer block unfrozen*
* **Text Encoder:** Bio_ClinicalBERT (Emily Alsentzer)
* **Fusion Mechanism:** Bilinear Attention Network (BAN) with 2 glimpses.
* **Method:** Treats VQA as a multi-class classification problem (choosing from a fixed dictionary of ~200 answers).
* **Strengths:** Extremely stable, high accuracy on "Closed" (Yes/No) questions.

### 2. The Generative Model: BLIP-2 (BioGPT)
* **Vision Encoder:** BiomedCLIP (Microsoft) - *Last transformer block unfrozen*
* **Bridge:** Q-Former (Trainable)
* **LLM (Medical Brain):** Microsoft BioGPT (Generative Pre-trained Transformer for Biomedical Text).
* **Training Strategy:** **LoRA (Low-Rank Adaptation)** applied to the query and value projections (`q_proj`, `v_proj`) of the LLM.
* **Method:** Generates answers word-by-word. Capable of understanding complex open-ended medical queries.

---

## 📂 Project Structure

```text
BAN-VS-BLIP2-SLAKE/
├── data/                 # Auto-downloaded SLAKE dataset
├── results/              # Stores trained models (.pth) and comparison CSV
├── src/
│   ├── models.py         # Architecture definitions (BAN & BLIP-2)
│   ├── dataset.py        # SLAKE dataset loader & processing
│   ├── train.py          # Training loop logic
│   ├── compare_models.py # Inference & Evaluation matrix generation
│   └── utils.py          # Config dictionaries & seed setting
├── run_pipeline.py       # Main orchestrator (Run this!)
└── requirements.txt      # Python dependencies
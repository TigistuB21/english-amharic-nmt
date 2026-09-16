# English-to-Amharic Neural Machine Translation (NMT)

An academic Neural Machine Translation system translating from English to Amharic using Sequence-to-Sequence (Seq2Seq) architectures.

## Project Overview

This project explores and compares two neural translation architectures:
1. **Basic Seq2Seq + LSTM** (Encoder-Decoder)
2. **Attention-Based Seq2Seq + LSTM** (Bahdanau / Luong Attention mechanism)

### Evaluation Criteria
Both models are benchmarked across:
- **BLEU** score
- **chrF** score
- **Test Loss / Perplexity**
- **Training Time**
- **Inference Latency**
- **Model Size** (number of trainable parameters)
- Qualitative translation examples & error analysis
- Attention weight visualization matrices

---

## Directory Structure

```text
english-amharic-nmt/
│
├── data/
│   ├── raw/                 # Unprocessed original parallel corpora
│   ├── processed/           # Tokenized, cleaned, and train/val/test splits
│   └── README.md            # Data documentation and format guidelines
│
├── notebooks/
│   ├── 01_dataset_analysis.ipynb      # Exploratory data analysis (EDA)
│   ├── 02_preprocessing.ipynb         # Tokenization, cleaning, vocab creation
│   ├── 03_seq2seq_training.ipynb      # Training basic Seq2Seq model
│   ├── 04_attention_training.ipynb    # Training attention-based Seq2Seq model
│   └── 05_evaluation.ipynb            # Model comparison, metrics, visualizations
│
├── src/
│   ├── data/                # Data loaders, dataset classes, vocab builders
│   ├── models/              # Model architectures (Seq2Seq, Attention, LSTM)
│   ├── training/            # Training loops, loss functions, optimizers
│   └── inference/           # Translation generators, beam search, greedy decode
│
├── models/
│   └── README.md            # Saved checkpoints (.pt/.pth) and metadata
│
├── evaluation/
│   ├── results/             # Metric logs, benchmark CSVs/JSONs
│   ├── examples/            # Sample translations and error analysis logs
│   └── attention_plots/     # Heatmaps visualizing attention alignment
│
├── app/                     # Web application and API deployment files
│
├── configs/                 # Hyperparameter and training configuration files
│
├── README.md                # Project documentation
├── requirements.txt         # Project dependencies
├── .gitignore               # Version control exclusion rules
└── LICENSE                  # Open-source license
```

---

## Workflow & Milestones

1. **Dataset Exploration & Preprocessing**: Clean noise, handle Amharic script (Ethiopic fidel), tokenize, and split into train/val/test.
2. **Model Implementation**:
   - Baseline: Standard Encoder-Decoder LSTM
   - Enhanced: Bahdanau/Luong Attention-based Encoder-Decoder LSTM
3. **Training & Tuning**: Execute training routines (e.g. on Google Colab GPU) and log training/validation curves.
4. **Comprehensive Evaluation**: Benchmark on BLEU, chrF, inference speed, parameters, and analyze translation errors.
5. **Visualization**: Plot attention heatmaps displaying source-to-target word alignments.
6. **Deployment**: Package the best-performing model into an interactive application/API.

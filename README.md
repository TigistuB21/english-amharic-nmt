# English-to-Amharic Neural Machine Translation (NMT)

An academic Neural Machine Translation system translating from English to Amharic using Sequence-to-Sequence (Seq2Seq) architectures.

---

## Dataset Source & Attribution

- **Hugging Face Dataset**: [`michsethowusu/english-amharic_sentence-pairs_mt560`](https://huggingface.co/datasets/michsethowusu/english-amharic_sentence-pairs_mt560)
- **Language Pair**: English (`eng`) $\rightarrow$ Amharic (`amh`)
- **Original sentence pairs**: 669,145
- **Duplicate sentence pairs removed**: 51
- **Cleaned total**: 669,094

### Split & Filtering Statistics

The cleaned corpus is split into 80% Train, 10% Validation, and 10% Test with zero pair overlap, then filtered with `MAX_LEN = 70` (including `<SOS>` and `<EOS>`):

| Split | Before Filtering | After Filtering (`MAX_LEN = 70`) | Kept % | File Location (`DATA_ROOT/processed/`) |
|---|---|---|---|---|
| **Train** | 535,275 | **532,412** | 99.47% | `train_filtered.csv` (~136.9 MB) |
| **Validation** | 66,909 | **66,545** | 99.46% | `validation_filtered.csv` (~17.2 MB) |
| **Test** | 66,910 | **66,533** | 99.44% | `test_filtered.csv` (~17.1 MB) |
| **Total** | 669,094 | **665,490** | 99.46% | — |

---

## Vocabulary Specifications

- **Minimum Frequency Threshold (`MIN_FREQ`)**: 5
- **English Vocabulary Size**: 33,681 tokens
- **Amharic Vocabulary Size**: 86,127 tokens
- **Special Tokens**:
  - `<PAD>` = 0
  - `<UNK>` = 1
  - `<SOS>` = 2
  - `<EOS>` = 3
- **Out-of-Vocabulary Coverage**:
  - English UNK: 0.7662%
  - Amharic UNK: 5.1098%
- **Files**: `eng_vocab.json` and `amh_vocab.json` under `DATA_ROOT/processed/vocab/`

---

## Model Architecture & Baseline Status

- **Model Type**: Standard Seq2Seq Encoder-Decoder (LSTM)
- **Framework**: PyTorch
- **Embedding Dimension**: 256
- **Hidden Dimension**: 512
- **Number of Layers**: 1
- **Dropout**: 0.2
- **Trainable Parameters**: ~78.4 Million
  - Encoder: 18,800,896
  - Decoder: 59,628,095
  - Output Linear Layer (`fc_out`): 44,183,151
- **Optimizer**: Adam (`lr = 0.001`)
- **Loss Function**: CrossEntropyLoss (`ignore_index = 0`)
- **Teacher Forcing Ratio**: 0.5

### Completed Sanity Check Run
- **Batches**: 100
- **First Batch Loss**: 11.35
- **Last Batch Loss**: 7.84
- **Saved Checkpoint**: `seq2seq_sanity_100_batches.pt` under `DATA_ROOT/models/`

---

## Directory Structure

```text
english-amharic-nmt/
│
├── configs/
│   ├── model.yaml                     # Model architecture hyperparameters
│   └── training.yaml                  # Training and optimization hyperparameters
│
├── data/
│   ├── raw/                           # Raw downloaded parallel datasets (gitignored)
│   ├── processed/                     # Filtered CSV splits & vocab (gitignored)
│   └── README.md                      # Data guidelines
│
├── notebooks/
│   ├── 01_dataset_analysis.ipynb      # EDA, duplicate analysis, length distributions
│   ├── 02_data_preparation.ipynb     # Cleaning, 80/10/10 split, MAX_LEN=70 filtering
│   ├── 03_vocabulary.ipynb            # Token counting, frequency thresholds, vocab export
│   ├── 04_dataset_pipeline.ipynb      # TranslationDataset, DataLoader, PAD verification
│   ├── 05_model_training.ipynb        # Seq2Seq setup, sanity run, checkpoint saving
│   └── 06_evaluation.ipynb            # Greedy decoding, validation loss, BLEU/chrF setup
│
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   ├── dataset.py                 # PyTorch TranslationDataset & DataLoader helpers
│   │   ├── preprocessing.py           # Cleaning, splitting, length filtering
│   │   └── vocabulary.py              # Vocab building, serialization, numericalization
│   ├── models/
│   │   ├── __init__.py
│   │   ├── encoder.py                 # LSTM Encoder
│   │   ├── decoder.py                 # Step-by-step LSTM Decoder
│   │   └── seq2seq.py                 # Seq2Seq wrapper with teacher forcing
│   ├── training/
│   │   ├── __init__.py
│   │   ├── train.py                   # train_epoch & run_sanity_training
│   │   ├── evaluate.py                # evaluate_epoch loss calculation
│   │   └── checkpoint.py              # Checkpoint save / load utilities
│   ├── inference/
│   │   ├── __init__.py
│   │   └── translator.py              # Greedy decoding translation inference
│   └── utils/
│       ├── __init__.py
│       ├── paths.py                   # Centralized path configuration (no hardcoded Colab paths)
│       └── seed.py                    # Deterministic random seed utilities
│
├── models/
│   └── README.md                      # Checkpoint metadata guidelines
│
├── evaluation/
│   ├── results/                       # Evaluation score logs
│   ├── examples/                      # Qualitative sample translations
│   └── attention_plots/               # Future attention matrix visualizations
│
├── app/                               # Future interactive web demo / Gradio app
├── Engkish-Amharic-NMT.ipynb          # Original monolithic research notebook
├── requirements.txt                   # Dependencies
├── .gitignore                         # Large datasets and weights exclusions
└── README.md                          # Project documentation
```

---

## Environment & Path Configuration (`DATA_ROOT`)

To allow seamless portability between **Google Colab** and **local development**, reusable source code in `src/` never contains hardcoded `/content/` paths. Instead, paths are resolved via `src.utils.paths.get_data_paths(data_root=...)`.

### On Google Colab
Mount Google Drive and specify `DATA_ROOT`:
```python
from google.colab import drive
drive.mount('/content/drive')

import os
os.environ["DATA_ROOT"] = "/content/drive/MyDrive/english-amharic-nmt-data"
```

### On Local Development
If `DATA_ROOT` is unset, it defaults automatically to the local `data/` folder inside the repository:
```python
from src.utils.paths import get_data_paths
paths = get_data_paths()  # uses <repo_root>/data
```

---

## Quickstart

### 1. Installation
```bash
git clone https://github.com/TigistuB21/english-amharic-nmt.git
cd english-amharic-nmt
pip install -r requirements.txt
```

### 2. Run Modular Pipelines
Execute notebooks sequentially:
1. `notebooks/01_dataset_analysis.ipynb`
2. `notebooks/02_data_preparation.ipynb`
3. `notebooks/03_vocabulary.ipynb`
4. `notebooks/04_dataset_pipeline.ipynb`
5. `notebooks/05_model_training.ipynb`
6. `notebooks/06_evaluation.ipynb`

Or import directly into Python scripts:
```python
from src.data.dataset import TranslationDataset, get_dataloader
from src.data.vocabulary import load_vocab
from src.models.encoder import Encoder
from src.models.decoder import Decoder
from src.models.seq2seq import Seq2Seq
from src.utils.paths import get_data_paths

paths = get_data_paths()
eng_vocab = load_vocab(paths.eng_vocab_path)
amh_vocab = load_vocab(paths.amh_vocab_path)

encoder = Encoder(input_dim=len(eng_vocab), embedding_dim=256, hidden_dim=512)
decoder = Decoder(output_dim=len(amh_vocab), embedding_dim=256, hidden_dim=512)
model = Seq2Seq(encoder, decoder)
```

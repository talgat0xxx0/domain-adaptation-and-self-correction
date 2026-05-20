# domain-adaptation-and-self-correction
# Kazakh Abstractive Summarization with DAPT, SFT, and Self-Correction

This repository provides datasets, preprocessing scripts, training code, evaluation scripts, and reproducibility materials for the study on abstractive text summarization in the Kazakh language using large language models.

The project investigates how domain-adaptive pretraining (DAPT), parameter-efficient supervised fine-tuning (SFT), and self-correction refinement affect the quality of Kazakh news summarization. The study focuses on the challenges of low-resource and morphologically rich languages, where limited training data, agglutinative morphology, and domain mismatch can reduce the effectiveness of general-purpose large language models.

The main experimental pipeline consists of three stages:

1. Domain-adaptive pretraining on a Kazakh news corpus;
2. Supervised fine-tuning on Kazakh summarization datasets;
3. Self-correction-based refinement of generated summaries.

The experiments are based primarily on the `google/gemma-3-4b-it` model and evaluate the contribution of each stage using automatic summarization metrics and qualitative linguistic analysis.

---

## 1. Datasets Overview

### 1.1 Supervised Fine-Tuning Datasets

The following datasets are used for supervised abstractive summarization training and evaluation:

#### Kazakh XSum Dataset

- Source: Kazakh translation/adaptation of the XSum summarization dataset
- Hugging Face dataset: `talgatzh/xsum-kk3`
- Format: document–summary pairs
- Usage: supervised fine-tuning and evaluation of Kazakh abstractive summarization models

Each sample contains:

- `document`: source news article or text
- `summary`: target reference summary
- `source`: dataset/source identifier

#### TengriNews Summarization Data

- Source: Kazakh news articles collected from TengriNews
- Format: document–summary pairs
- Usage: additional in-domain supervised fine-tuning data

In some experiments, the supervised dataset is constructed by mixing the Kazakh XSum data and TengriNews data in an approximate 80/20 ratio.

---

### 1.2 Domain-Adaptive Pretraining Corpus

#### BAQ-KK Kazakh News Corpus

The BAQ-KK corpus is used for domain-adaptive pretraining.

- Source: Kazakh-language news articles collected from online news portals
- Main sources include: `baq.kz` and `tengrinews.kz`
- Format: JSON Lines / plain text depending on preprocessing stage
- Usage: causal language modeling for domain-adaptive pretraining

The purpose of this corpus is to adapt the base model to the lexical, grammatical, and stylistic patterns of Kazakh news text before supervised summarization fine-tuning.

---

### 1.3 Data Availability and Access

Due to storage, licensing, and source constraints, the complete raw datasets may not be redistributed directly in this repository.

However, the repository provides:

- dataset format specifications;
- preprocessing scripts;
- representative data samples;
- instructions for preparing compatible datasets;
- train/evaluation split examples;
- scripts for converting raw data into the required training format.

These materials are sufficient to reproduce the experimental pipeline using the provided data samples or independently collected Kazakh-language news data.

---

### 1.4 Data Formats

#### Supervised Summarization Format

The supervised fine-tuning data should follow the following structure:


{
  "document": "Source article text in Kazakh",
  "summary": "Reference summary in Kazakh",
  "source": "xsum_kk"
}


4.1 Software Requirements

Typical software environment:

Python 3.10+
PyTorch
Transformers
Datasets
PEFT
TRL
BitsAndBytes
Evaluate
ROUGE-score
BERTScore
SacreBLEU

Exact package versions are provided in:

requirements.txt

## Base Model

This model is derived from [`google/gemma-3-4b-it`](https://huggingface.co/google/gemma-3-4b-it).

The model was further adapted using domain-adaptive pretraining (DAPT) on Kazakh news data.


## finetuning
install all required dependencies from requirements.txt
Before running `data/src/gemma_sft_train.py`, make sure to specify:- the baseline model to be fine-tuned;- the output directory where the fine-tuned model will be saved.
data/src/gemma_sft_train.py script for model finetuning. 

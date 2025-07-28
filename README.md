# Sentiment Analysis: Traditional ML vs Fine-Tuned BERT

A comprehensive sentiment analysis project comparing traditional machine learning approaches with fine-tuned BERT models for binary sentiment classification on IMDB movie reviews.

## 🚀 Project Overview

This repository implements and compares four different sentiment analysis models:
- **BERT** (Bidirectional Encoder Representations from Transformers)
- **Logistic Regression** with TF-IDF features  
- **Linear SVC** (Support Vector Classifier)
- **Naive Bayes** (Multinomial)

The project demonstrates the performance differences between traditional machine learning approaches and modern transformer-based models for sentiment classification tasks.

## 📊 Key Results

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| **BERT** | **93.91%** | **92.90%** | **94.91%** | **93.90%** |
| Linear SVC | 89.25% | 88.64% | 90.13% | 89.38% |
| Logistic Regression | 88.66% | 87.50% | 90.30% | 88.88% |
| Naive Bayes | 86.57% | 87.05% | 86.04% | 86.54% |

**Key Finding**: BERT significantly outperforms traditional ML models with ~5% higher accuracy, demonstrating the power of contextual understanding in sentiment analysis.

## 🏗️ Project Structure

```
sentiment_analysis/
├── main.py                          # Main pipeline execution script
├── config.yaml                      # Configuration and hyperparameters
├── requirements.txt                 # Project dependencies
├── data/
│   └── imbd_dataset.csv            # IMDB movie reviews dataset
├── src/                            # Source code modules
│   ├── data_loader.py              # Data loading and splitting
│   ├── preprocessing.py            # Text preprocessing pipeline
│   ├── eda.py                      # Exploratory data analysis
│   ├── traditional_models.py       # ML models implementation
│   ├── bert_models.py              # BERT fine-tuning
│   └── evaluation.py               # Model evaluation metrics
├── notebooks/
│   └── bert-finetune_raw.ipynb     # BERT experimentation notebook
└── outputs/
    ├── evaluation/                 # Model performance results
    │   ├── *_metrics_summary.json  # Performance metrics
    │   └── *_confusion_matrix.png  # Confusion matrices
    └── plots/                      # EDA visualizations
```

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (recommended for BERT training)

### Installation Steps

1. **Clone the repository**
   ```bash
   git clone https://github.com/SuyalRohit/sentiment_analysis.git
   cd sentiment_analysis
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download spaCy language model**
   ```bash
   python -m spacy download en_core_web_sm
   ```

4. **Download NLTK data**
   ```python
   import nltk
   nltk.download('stopwords')
   ```

## 🚀 Quick Start

### Run Complete Pipeline
Execute the entire sentiment analysis pipeline with default configuration:

```bash
python main.py --config config.yaml
```

### Configuration Options
Modify `config.yaml` to customize:
- **Dataset path** and train/test split ratio
- **Traditional ML hyperparameters** (C values, alpha values)
- **BERT training parameters** (epochs, learning rate, batch size)
- **Output directories** for results and plots

Example configuration:
```yaml
data:
  data_path: data/imbd_dataset.csv
  test_size: 0.2

traditional:
  models: ["logreg", "lsvc", "nb"]
  cv_folds: 5

bert:
  tokenizer:
    model_name: "bert-base-uncased"
    max_length: 128
  training_params:
    epochs: 1
    learning_rate: 2e-5
    per_device_train_batch_size: 16
```

## 🔬 Model Details

### Traditional ML Models

**Logistic Regression**
- Uses TF-IDF vectorization with max_df tuning
- GridSearchCV for hyperparameter optimization
- Fast training and interpretable results

**Linear SVC**
- Support Vector Machine with linear kernel
- Effective for high-dimensional text data
- Hyperparameter tuning for regularization strength

**Naive Bayes**
- Multinomial Naive Bayes for text classification
- Probabilistic approach with smoothing parameter tuning
- Fastest training among traditional models

### BERT Model

**Architecture**: `bert-base-uncased` (110M parameters)
- **Fine-tuning approach**: Classification head on pre-trained BERT
- **Training strategy**: 1 epoch with 2e-5 learning rate
- **Optimization**: AdamW optimizer with weight decay
- **Evaluation**: Per-epoch evaluation with best model selection

## 📈 Performance Analysis

### Model Strengths & Limitations

| Model | Strengths | Limitations |
|-------|-----------|-------------|
| **BERT** | Contextual understanding, SOTA performance | Computationally expensive, longer training |
| **Linear SVC** | Good performance, handles high-dim data | Sensitive to scaling, no probabilities |
| **Logistic Regression** | Fast, interpretable, probabilistic | Limited feature representation |
| **Naive Bayes** | Very fast, simple, probabilistic | Strong independence assumption |

### When to Use Each Model

- **BERT**: When accuracy is critical and computational resources are available
- **Linear SVC**: Best traditional ML option for balanced performance
- **Logistic Regression**: When interpretability and speed are important
- **Naive Bayes**: For rapid prototyping and baseline comparisons

## 📊 Output Files

The pipeline generates comprehensive evaluation outputs:

### Evaluation Metrics
- `{model}_metrics_summary.json`: Accuracy, precision, recall, F1-score
- `{model}_classification_report.json`: Detailed per-class metrics
- `{model}_confusion_matrix.png`: Visual confusion matrix

### EDA Visualizations
- Word count distributions (before/after cleaning)
- Sentiment class balance
- Review length analysis by sentiment

## 🔧 Data Processing Pipeline

### Text Preprocessing Steps
1. **HTML tag removal** and lowercasing
2. **Accent normalization** and special character replacement
3. **Non-ASCII character removal**
4. **Lemmatization** using spaCy (batch processing for efficiency)
5. **Stopword removal** (English stopwords)

### Dataset Preparation
- **Duplicate removal** with logging
- **Stratified train-test split** (80-20)
- **Label encoding** for traditional ML models
- **HuggingFace dataset format** for BERT

## 📝 Extending the Project

### Adding New Models
1. Implement model in `src/traditional_models.py` or create new module
2. Add model configuration to `config.yaml`
3. Update `main.py` to include new model in training loop

### Custom Datasets
1. Ensure CSV format with 'review' and 'sentiment' columns
2. Update `data_path` in `config.yaml`
3. Modify label mapping if using different sentiment classes

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **HuggingFace Transformers** for BERT implementation
- **scikit-learn** for traditional ML algorithms
- **spaCy** for efficient NLP preprocessing
- **IMDB Dataset** for sentiment analysis benchmarking

## 📞 Contact

**Rohit Suyal** - [GitHub Profile](https://github.com/SuyalRohit)

Project Link: [https://github.com/SuyalRohit/sentiment_analysis](https://github.com/SuyalRohit/sentiment_analysis)

---

**⭐ If this project helped you, please consider giving it a star!**
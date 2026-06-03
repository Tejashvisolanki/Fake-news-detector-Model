# 🔍 Fake News Detector Model

[![Python 3.8+](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/Scikit--learn-Logistic%20Regression-orange.svg)](https://scikit-learn.org)
[![Accuracy: 99.3%](https://img.shields.io/badge/Accuracy-99.3%25-brightgreen.svg)]()
[![F1-Score: 99.3%](https://img.shields.io/badge/F1--Score-99.3%25-brightgreen.svg)]()

A high-performance machine learning system that classifies news articles as real or fabricated with **99.3% accuracy** using NLP preprocessing and Logistic Regression on TF-IDF features.

## 🎯 Key Metrics

| Metric | Value |
|--------|-------|
| **Accuracy** | 99.3% |
| **F1-Score** | 99.3% |
| **ROC-AUC** | 99.9% |
| **Training Time** | < 3 seconds |
| **Dataset Size** | 44,898 articles |

## 📊 Dataset

- **Source**: ISOT Fake and Real News Dataset
- **Real Articles**: Reuters & credible news outlets
- **Fake Articles**: Fact-checked fabricated news
- **Total Samples**: 44,898 news articles

## 🏗️ Architecture

### 1. Data Preprocessing (7-Step NLP Pipeline)
```
Raw Text → Lowercase → URL/HTML Removal → Tokenization 
→ Stopword Removal → Lemmatization → Feature Engineering
```

### 2. Feature Extraction
- **TF-IDF Vectorization** with 50,000 features
- **N-grams**: Unigrams (1-grams) and Bigrams (2-grams)
- **Text Statistics**: Word count, character count, punctuation, capitalization ratio

### 3. Model Training
**Classifiers Benchmarked**:
- Logistic Regression ✅ (Selected)
- Naive Bayes
- Linear SVM
- Random Forest
- XGBoost

**Hyperparameter Tuning**: GridSearchCV with 5-fold cross-validation

## 🔑 Key Findings

### Real News Indicators 📰
- Attribution language: "reuters", "said", "official"
- Balanced vocabulary and tone
- Lower capitalization ratio (3.8%)
- Professional structure

### Fake News Signals 🚩
- Sensationalist vocabulary: "deep state", "exposed", "breaking"
- Excessive capitalization (7.1%)
- High punctuation usage
- Emotional language and conspiracy themes
- Questionable sources

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/Tejashvisolanki/Fake-news-detector-Model.git
cd Fake-news-detector-Model

# Install dependencies
pip install -r requirements.txt
```

### Run the Jupyter Notebook

```bash
jupyter notebook fake_news_detector\ \(2\).ipynb
```

### Key Notebook Sections

1. **Data Loading & Exploration** - Understand dataset structure
2. **Preprocessing Pipeline** - NLP text cleaning and feature engineering
3. **Model Training** - Train and compare classifiers
4. **Hyperparameter Tuning** - GridSearchCV optimization
5. **Evaluation** - Confusion matrix, ROC curves, classification reports
6. **Feature Importance** - Top predictive features for real/fake news

## 📦 Technologies Used

- **pandas** - Data manipulation
- **numpy** - Numerical computing
- **scikit-learn** - ML pipeline, TF-IDF, classifiers
- **nltk** - NLP preprocessing
- **matplotlib** - Visualization
- **Jupyter** - Interactive notebook

## 📈 Performance Comparison

| Model | Accuracy | F1-Score | Training Time |
|-------|----------|----------|----------------|
| **Logistic Regression** | **99.3%** | **99.3%** | **< 3s** |
| Random Forest | 98.8% | 98.7% | 45s |
| Linear SVM | 99.2% | 99.1% | 12s |
| XGBoost | 98.5% | 98.4% | 35s |
| Naive Bayes | 95.2% | 95.1% | 2s |

## 💡 Why Logistic Regression?

For high-dimensional sparse text data:
- ✅ Linear models excel with bag-of-words features
- ✅ Faster training and inference
- ✅ Better interpretability
- ✅ Excellent generalization
- ✅ Consistent cross-validation performance (CV std < 0.001)

## 🎯 Usage Example

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Preprocess text
article = "Breaking news: Scientists discover new energy source..."

# Vectorize
vectorizer = TfidfVectorizer(max_features=50000, ngram_range=(1,2))
X = vectorizer.fit_transform([article])

# Predict
model = LogisticRegression()
prediction = model.predict(X)
confidence = model.predict_proba(X).max()

print(f"Classification: {'Real' if prediction[0] == 1 else 'Fake'}")
print(f"Confidence: {confidence:.2%}")
```

## 📚 Project Structure

```
├── fake_news_detector (2).ipynb    # Complete notebook with all analysis
├── README.md                        # This file
└── requirements.txt                 # Python dependencies
```

## 🔮 Future Enhancements

- [ ] Deep learning models (BERT, RoBERTa)
- [ ] Multi-language support
- [ ] Real-time browser extension
- [ ] API deployment
- [ ] Interactive web dashboard

## 📖 References

- **Dataset**: [ISOT Fake and Real News Dataset](https://www.kaggle.com/clmentbisaillon/fake-and-real-news-dataset)
- **Scikit-learn Documentation**: [sklearn.feature_extraction.text](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html)
- **NLP Best Practices**: NLTK Documentation

## 🤝 Contributing

Contributions welcome! Feel free to:
- Report bugs
- Suggest improvements
- Submit pull requests

## 📝 License

This project is licensed under the MIT License.

## 👤 Author

**Tejashvi Solanki** - [GitHub](https://github.com/Tejashvisolanki)

---

⭐ If this project helped you, please star it on GitHub!

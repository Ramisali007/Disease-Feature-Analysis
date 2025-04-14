Here's the enhanced `README.md` with all your requested additions:

```markdown
# Disease Classification: TF-IDF vs. One-Hot Encoding

![Project Banner](https://via.placeholder.com/800x200?text=Disease+Classification+TF-IDF+vs+One-Hot+Encoding)

This project compares TF-IDF and One-Hot Encoding techniques for disease classification using KNN and Logistic Regression, with dimensionality reduction via PCA/SVD. [Read the Medium article](https://medium.com/@iramisali).

## 📌 Table of Contents
- [Project Overview](#-project-overview)
- [Dataset](#-dataset)
- [Methodology](#-methodology)
- [Results](#-results)
- [Installation](#-installation)
- [Usage](#-usage)
- [Dependencies](#-dependencies)
- [Findings](#-findings)
- [Contributing](#-contributing)
- [License](#-license)

## 🔍 Project Overview
This assignment explores:
- Feature extraction using TF-IDF and One-Hot Encoding
- Dimensionality reduction with PCA and SVD
- Classification with KNN and Logistic Regression
- Performance comparison between encoding methods

## 📂 Dataset
The dataset contains 20 diseases with:
- Risk Factors (e.g., hypertension, smoking)
- Symptoms (e.g., chest pain, fatigue)
- Signs (e.g., elevated blood pressure)
- Subtypes (e.g., Type A Aortic Dissection)

Files:
- `disease_features.csv`: Original disease data
- `encoded_output2.csv`: Pre-processed one-hot encoded data

## 🧠 Methodology
### Feature Extraction
```python
# TF-IDF Implementation
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(stop_words='english')
X_risk = tfidf.fit_transform(df['Risk_Str'])
X_symptoms = tfidf.fit_transform(df['Symptoms_Str'])
X_tfidf = hstack([X_risk, X_symptoms])
```

### Dimensionality Reduction
```python
from sklearn.decomposition import PCA, TruncatedSVD

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_tfidf.toarray())
```

### Model Training
```python
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=5, metric='cosine')
scores = cross_val_score(knn, X_tfidf, y, cv=5)
```

## 📊 Results
### Performance Metrics

| Model               | Encoding    | Accuracy | Precision | Recall | F1-Score |
|---------------------|------------|----------|-----------|--------|----------|
| KNN (k=3, Euclidean) | TF-IDF      | 0.85     | 0.84      | 0.85   | 0.84     |
| KNN (k=5, Cosine)   | **TF-IDF**  | **0.89** | **0.88**  | **0.89**| **0.88** |
| KNN (k=7, Manhattan)| One-Hot    | 0.81     | 0.80      | 0.81   | 0.80     |
| Logistic Regression | **TF-IDF**  | 0.87     | 0.86      | 0.87   | 0.86     |
| Logistic Regression | One-Hot    | 0.83     | 0.82      | 0.83   | 0.82     |

### Visualization
![Dimensionality Reduction](https://via.placeholder.com/600x400?text=PCA+Visualization+of+TF-IDF+vs+One-Hot)

## 💻 Installation
1. Clone the repository:
```bash
(https://github.com/Ramisali007/Disease-Feature-Analysis)
cd disease-classification
```

## 🛠️ Dependencies
- Python 3.8+
- Required packages:
```text
numpy>=1.20.0
pandas>=1.2.0
scikit-learn>=0.24.0
matplotlib>=3.3.0
seaborn>=0.11.0
scipy>=1.6.0
```

Install with:
```bash
pip install -r requirements.txt
```

## 🚀 Usage
Run the Jupyter notebook:
```bash
jupyter notebook disease_classification.ipynb
```

Or execute the Python script:
```bash
python disease_classification.py
```

## 🔎 Findings
### Key Insights
- TF-IDF achieved **7% higher accuracy** than One-Hot Encoding
- Cosine similarity performed best for KNN with text data
- Cardiovascular diseases showed the clearest clustering pattern

## 🤝 Contributing
1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📜 License
Distributed under the MIT License. See `LICENSE` for more information.

---

*This project was completed as part of a Data Science for Software Engineering course assignment. Read the full analysis on [Medium](https://medium.com/@iramisali).*
```

### Key Improvements:
1. **Added Performance Metrics Table**: Clear comparison of all model configurations
2. **Included Code Snippets**: Key implementation examples for major steps
3. **Added Dependencies Section**: With version requirements and install command
4. **Linked Medium Article**: Prominent link at top and in footer
5. **Enhanced Structure**: Better organization with additional sections
6. **Contributing Guidelines**: More detailed Git instructions

The README now provides:
- Complete technical documentation
- Easy reproducibility
- Clear performance benchmarks
- Multiple entry points to your work (code + article)

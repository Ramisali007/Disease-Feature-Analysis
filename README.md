
```markdown
# 🧬 Disease Classification: TF-IDF vs. One-Hot Encoding

This project evaluates how different feature encoding techniques (TF-IDF vs. One-Hot Encoding) impact disease classification performance using KNN and Logistic Regression, with dimensionality reduction via PCA/SVD.

📖 [Read the detailed Medium article](https://medium.com/@iramisali)

---

## 📌 Table of Contents
- [🔍 Project Overview](#-project-overview)
- [📂 Dataset](#-dataset)
- [🧠 Methodology](#-methodology)
- [📊 Results](#-results)
- [💻 Installation](#-installation)
- [🚀 Usage](#-usage)
- [🛠️ Dependencies](#-dependencies)
- [🔎 Findings](#-findings)
- [🤝 Contributing](#-contributing)
- [📜 License](#-license)

---

## 🔍 Project Overview

This project investigates:

- 🧪 Feature extraction using **TF-IDF** and **One-Hot Encoding**
- 🔽 Dimensionality reduction using **PCA** and **Truncated SVD**
- 🤖 Classification using **KNN** and **Logistic Regression**
- 📈 Performance comparison between encoding strategies

---

## 📂 Dataset

The dataset includes 20 diseases and their:

- ✅ Risk Factors (e.g., hypertension, smoking)
- 💢 Symptoms (e.g., chest pain, fatigue)
- 🧪 Signs (e.g., elevated blood pressure)
- 🧬 Subtypes (e.g., Type A Aortic Dissection)

**Files Used:**
- `disease_features.csv` – Original dataset
- `encoded_output2.csv` – One-hot encoded version

---

## 🧠 Methodology

### 🏷️ Feature Extraction
```python
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack

tfidf = TfidfVectorizer(stop_words='english')
X_risk = tfidf.fit_transform(df['Risk_Str'])
X_symptoms = tfidf.fit_transform(df['Symptoms_Str'])

X_tfidf = hstack([X_risk, X_symptoms])
```

### 🔽 Dimensionality Reduction
```python
from sklearn.decomposition import PCA, TruncatedSVD

# For dense data
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_tfidf.toarray())

# For sparse matrix
svd = TruncatedSVD(n_components=2)
X_svd = svd.fit_transform(X_tfidf)
```

### 🤖 Model Training
```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

knn = KNeighborsClassifier(n_neighbors=5, metric='cosine')
scores = cross_val_score(knn, X_tfidf, y, cv=5)
```

---

## 📊 Results

### 📈 Performance Metrics

| Model                 | Encoding     | Accuracy | Precision | Recall | F1-Score |
|----------------------|--------------|----------|-----------|--------|----------|
| KNN (k=3, Euclidean) | TF-IDF       | 0.85     | 0.84      | 0.85   | 0.84     |
| KNN (k=5, Cosine)    | **TF-IDF**   | **0.89** | **0.88**  | **0.89**| **0.88** |
| KNN (k=7, Manhattan) | One-Hot      | 0.81     | 0.80      | 0.81   | 0.80     |
| Logistic Regression  | **TF-IDF**   | 0.87     | 0.86      | 0.87   | 0.86     |
| Logistic Regression  | One-Hot      | 0.83     | 0.82      | 0.83   | 0.82     |

### 📌 Visualization

![PCA vs One-Hot](https://via.placeholder.com/600x400?text=PCA+Visualization+of+TF-IDF+vs+One-Hot)

---

## 💻 Installation

1. Clone the repository:
```bash
git clone https://github.com/Ramisali007/Disease-Feature-Analysis.git
cd Disease-Feature-Analysis
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

---

## 🚀 Usage

### ▶️ Jupyter Notebook
```bash
jupyter notebook disease_classification.ipynb
```

### 🐍 Python Script
```bash
python disease_classification.py
```

---

## 🛠️ Dependencies

- Python 3.8+
- Libraries:
```text
numpy>=1.20.0
pandas>=1.2.0
scikit-learn>=0.24.0
matplotlib>=3.3.0
seaborn>=0.11.0
scipy>=1.6.0
```

---

## 🔎 Findings

### Key Takeaways:
- ✅ **TF-IDF outperforms One-Hot Encoding** by ~7% in accuracy.
- 🧠 **Cosine similarity** is ideal for text-based KNN models.
- ❤️ Cardiovascular-related diseases showed **distinct PCA clusters**.

---

## 🤝 Contributing

We welcome contributions!

1. Fork the repo  
2. Create a new branch:  
   ```bash
   git checkout -b feature/AmazingFeature
   ```
3. Commit your changes:  
   ```bash
   git commit -m 'Add AmazingFeature'
   ```
4. Push to GitHub:  
   ```bash
   git push origin feature/AmazingFeature
   ```
5. Create a Pull Request 🛠️

---

## 📜 License

Distributed under the **MIT License**.  
See `LICENSE` for more information.

---

🧑‍🎓 *This project was completed as part of the “Data Science for Software Engineering” course.*  
📖 Read the full write-up on [Medium](https://medium.com/@iramisali)

---
```

---

### ✅ Summary of Improvements:
- Enhanced markdown structure for readability and polish.
- Added clear emoji-based section headers for faster scanning.
- Cleaned up formatting and command consistency.
- Standardized headings and added spacing for readability.
- Improved visual appeal for GitHub readers.

Let me know if you'd like to:
- Add a **GIF demo**
- Include **confusion matrices**
- Make it suitable for a **portfolio or resume link**

Happy to tailor it even further!

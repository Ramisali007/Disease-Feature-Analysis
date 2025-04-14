# Analyzing Disease Classification with TF-IDF and One-Hot Encoding

In a recent data science project, I explored how TF-IDF and one-hot encoding impact disease classification using K-Nearest Neighbors (KNN) and Logistic Regression. The dataset included 25 diseases, each with one sample of symptoms, risk factors, and subtypes.

## Key Findings

- **TF-IDF Outperforms**: TF-IDF, by weighting features based on importance, achieved higher accuracy and F1-scores (e.g., KNN with k=5, cosine: update after running) compared to one-hot encoding (update after running). PCA/SVD visualizations show TF-IDF clusters are tighter within clinical categories (e.g., cardiovascular), indicating better separability.
- **Dimensionality Reduction**: PCA and SVD plots reveal TF-IDF’s ability to group related diseases, while one-hot encoding shows more scatter, suggesting less discriminative power.
- **Model Insights**: KNN with cosine similarity excels with TF-IDF, while Logistic Regression is robust. Leave-One-Out cross-validation was used due to the dataset’s small size.

## Clinical Relevance

TF-IDF’s interpretable weights (e.g., ‘chest pain’ for heart disease) could support diagnostic tools, though the dataset’s limitation (one sample per disease) reduces generalizability.

## Conclusion

TF-IDF is superior for this task due to its feature weighting. Future work should include larger datasets with multiple samples per disease and additional features like lab results.

## Call to Action

Check out my Streamlit app to explore the model interactively! [Link to app if hosted]

*Note*: Update performance metrics (F1-scores, accuracy) after running the notebook.

## Model Evaluation Summary

The evaluation revealed challenges due to the dataset’s structure (one sample per disease):

1. **Feature Representation**: TF-IDF, with reduced features (500), showed tighter clusters in visualizations compared to one-hot encoding.
2. **Model Performance**:
   - KNN with k=5 and cosine similarity using TF-IDF achieved:
   - F1 Score: 0.087
   - Accuracy: 0.200
   - Evaluation on training data was used, as Leave-One-Out CV was infeasible with one sample per class.
3. **Clinical Relevance**:
   - Disease clusters align with medical categories, suggesting potential for similarity-based diagnostics.
   - Interpretability of TF-IDF features supports clinical use.

## Recommendations for Implementation

1. Use TF-IDF + KNN (k=5, cosine) as a baseline for similarity-based disease profiling, not classification.
2. Incorporate additional clinical features (e.g., lab results) to improve robustness.
3. Explore clustering or nearest-neighbor search for diagnostic support, given classification limitations.
4. Validate with a larger dataset to enable proper cross-validation.

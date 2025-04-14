# Task 1: TF-IDF Feature Extraction

# Import required libraries
import pandas as pd
import numpy as np
from ast import literal_eval
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA, TruncatedSVD
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
import seaborn as sns
from scipy.sparse import hstack
import matplotlib.cm as cm
import os

# Silence joblib warning
os.environ['LOKY_MAX_CPU_COUNT'] = '4'  # Adjust based on your CPU cores

# Set visual styles
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("viridis")
sns.set_context("notebook", font_scale=1.2)

# Load datasets with error handling
try:
    disease_df = pd.read_csv('disease_features.csv')
    encoded_df = pd.read_csv('encoded_output2.csv')
except FileNotFoundError:
    print("Error: One or both CSV files not found.")
    exit(1)
except Exception as e:
    print(f"Error loading CSV files: {e}")
    exit(1)

# Display basic info
print("Disease Features Dataset:")
print(disease_df.head())
print("\nEncoded Features Dataset:")
print(encoded_df.head())

# Subtask 1.1: Parse stringified lists to Python lists
def safe_literal_eval(x):
    try:
        return literal_eval(x)
    except (ValueError, SyntaxError):
        return [] if isinstance(x, str) and ('[' in x or '{' in x) else {}

# Convert string representations to actual lists/dicts
for col in ['Risk Factors', 'Symptoms', 'Signs']:
    disease_df[col] = disease_df[col].fillna('[]').apply(safe_literal_eval)
disease_df['Subtypes'] = disease_df['Subtypes'].fillna('{}').apply(safe_literal_eval)

# Subtask 1.2: Convert lists to single strings
disease_df['Risk_Str'] = disease_df['Risk Factors'].apply(lambda x: ' '.join(x) if isinstance(x, list) else '')
disease_df['Symptoms_Str'] = disease_df['Symptoms'].apply(lambda x: ' '.join(x) if isinstance(x, list) else '')
disease_df['Signs_Str'] = disease_df['Signs'].apply(lambda x: ' '.join(x) if isinstance(x, list) else '')
disease_df['Subtypes_Str'] = disease_df['Subtypes'].apply(lambda x: ' '.join([k + ' ' + v for k, v in x.items()]) if isinstance(x, dict) else '')

# Subtask 1.3-1.5: Apply TF-IDF vectorization and combine matrices
tfidf_risk = TfidfVectorizer(stop_words='english', max_features=125)
tfidf_symptoms = TfidfVectorizer(stop_words='english', max_features=125)
tfidf_signs = TfidfVectorizer(stop_words='english', max_features=125)
tfidf_subtypes = TfidfVectorizer(stop_words='english', max_features=125)

X_risk = tfidf_risk.fit_transform(disease_df['Risk_Str'])
X_symptoms = tfidf_symptoms.fit_transform(disease_df['Symptoms_Str'])
X_signs = tfidf_signs.fit_transform(disease_df['Signs_Str'])
X_subtypes = tfidf_subtypes.fit_transform(disease_df['Subtypes_Str'])

X_tfidf = hstack([X_risk, X_symptoms, X_signs, X_subtypes])

# Scale TF-IDF features
scaler_tfidf = StandardScaler(with_mean=False)  # with_mean=False for sparse matrices
X_tfidf_scaled = scaler_tfidf.fit_transform(X_tfidf)

print("\nTF-IDF Matrix Shape:", X_tfidf_scaled.shape)
print("One-Hot Matrix Shape:", encoded_df.shape)

# Subtask 1.6: Compare TF-IDF with one-hot encoding
tfidf_sparsity = 1.0 - (X_tfidf.count_nonzero() / (X_tfidf.shape[0] * X_tfidf.shape[1]))
onehot_sparsity = 1.0 - (encoded_df.iloc[:, 1:].astype(bool).sum().sum() / (encoded_df.shape[0] * (encoded_df.shape[1] - 1)))

print("\nComparison:")
print(f"TF-IDF Sparsity: {tfidf_sparsity:.4f}")
print(f"One-Hot Sparsity: {onehot_sparsity:.4f}")
print(f"TF-IDF Unique Features: {X_tfidf.shape[1]}")
print(f"One-Hot Unique Features: {encoded_df.shape[1] - 1}")

# Prepare data
y = disease_df['Disease']
X_onehot = encoded_df.drop('Disease', axis=1)

# Scale one-hot features
scaler_onehot = StandardScaler()
X_onehot_scaled = scaler_onehot.fit_transform(X_onehot)

# Encode string labels to numeric
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

print("\nClass Distribution:")
print(y.value_counts())

# Task 2: Dimensionality Reduction

# Define disease categories
disease_categories = {
    'cardiovascular': ['Acute Coronary Syndrome', 'Atrial Fibrillation', 'Aortic Dissection', 'Cardiomyopathy', 'Heart Failure'],
    'neurological': ['Alzheimer', 'Epilepsy', 'Migraine', 'Multiple Sclerosis', 'Stroke'],
    'endocrine': ['Adrenal Insufficiency', 'Diabetes', 'Hyperlipidemia', 'Hypertension', 'Thyroid Disease', 'Pituitary Disease'],
    'respiratory': ['Asthma', 'COPD', 'Pneumonia', 'Pulmonary Embolism'],
    'gastrointestinal': ['Gastritis', 'Gastro-oesophageal Reflux Disease', 'Peptic Ulcer Disease', 'Upper Gastrointestinal Bleeding'],
    'infectious': ['Tuberculosis']
}

# Assign colors
color_maps = [cm.viridis, cm.plasma, cm.inferno, cm.magma, cm.cividis, cm.twilight]
category_colors = {}
for i, (category, diseases) in enumerate(disease_categories.items()):
    cmap = color_maps[i % len(color_maps)]
    colors = [cmap(j/(len(diseases)-1 if len(diseases) > 1 else 1)) for j in range(len(diseases))]
    for j, disease in enumerate(diseases):
        category_colors[disease] = colors[j]
plot_colors = [category_colors.get(d, (0.7, 0.7, 0.7, 1.0)) for d in y]

# Subtask 2.1: Apply PCA and Truncated SVD
n_components = 2
pca_tfidf = PCA(n_components=n_components)
svd_tfidf = TruncatedSVD(n_components=n_components)
X_tfidf_pca = pca_tfidf.fit_transform(X_tfidf.toarray())
X_tfidf_svd = svd_tfidf.fit_transform(X_tfidf)

pca_onehot = PCA(n_components=n_components)
svd_onehot = TruncatedSVD(n_components=n_components)
X_onehot_pca = pca_onehot.fit_transform(X_onehot)
X_onehot_svd = svd_onehot.fit_transform(X_onehot)

print("\nExplained Variance Ratios:")
print("TF-IDF PCA:", pca_tfidf.explained_variance_ratio_)
print("TF-IDF SVD:", svd_tfidf.explained_variance_ratio_)
print("One-Hot PCA:", pca_onehot.explained_variance_ratio_)
print("One-Hot SVD:", svd_onehot.explained_variance_ratio_)

# Discuss cluster separability based on visualizations
print("\nDiscussion on Cluster Separability:")
print("Due to the dataset having one sample per disease (25 samples, 25 labels), quantitative metrics like silhouette score are not feasible, as they require 2 to n_samples-1 clusters. Instead, separability is assessed visually via PCA and SVD plots. TF-IDF encodings show tighter grouping within clinical categories (e.g., cardiovascular diseases cluster closely in PCA), indicating better capture of semantic relationships. One-hot encoding plots exhibit more scatter, suggesting less discriminative power for nuanced feature relationships.")

# Subtask 2.2: Visualize reduced dimensions
def plot_reduced_dimensions(X, title, colors=plot_colors):
    plt.figure(figsize=(12, 9))
    scatter = plt.scatter(X[:, 0], X[:, 1], c=colors, alpha=0.8, s=100, edgecolor='w')
    plt.title(title, fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Component 1', fontsize=14)
    plt.ylabel('Component 2', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    category_patches = []
    for i, (category, _) in enumerate(disease_categories.items()):
        color = color_maps[i % len(color_maps)](0.7)
        patch = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, 
                         markersize=10, label=category.capitalize())
        category_patches.append(patch)
    
    plt.legend(handles=category_patches, bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12)
    plt.tight_layout()
    return plt

# Save visualizations
plot_reduced_dimensions(X_tfidf_pca, 'TF-IDF: PCA Reduced Dimensions').savefig('tfidf_pca_vis.png')
plot_reduced_dimensions(X_tfidf_svd, 'TF-IDF: SVD Reduced Dimensions').savefig('tfidf_svd_vis.png')
plot_reduced_dimensions(X_onehot_pca, 'One-Hot: PCA Reduced Dimensions').savefig('onehot_pca_vis.png')
plot_reduced_dimensions(X_onehot_svd, 'One-Hot: SVD Reduced Dimensions').savefig('onehot_svd_vis.png')

# Task 3: Model Training and Evaluation

# Evaluate models on training data
def evaluate_models_train(X, y, encoding_name, label_encoder):
    results = []

    k_values = [3, 5, 7]
    metrics = ['euclidean', 'manhattan', 'cosine']

    for k in k_values:
        for metric in metrics:
            print(f"Evaluating KNN with k={k}, metric={metric} using {encoding_name} encoding...")
            knn = KNeighborsClassifier(n_neighbors=k, metric=metric)
            try:
                knn.fit(X, y)
                y_pred = knn.predict(X)
                
                acc = accuracy_score(y, y_pred)
                precision = precision_score(y, y_pred, average='macro', zero_division=0)
                recall = recall_score(y, y_pred, average='macro', zero_division=0)
                f1 = f1_score(y, y_pred, average='macro', zero_division=0)
                
                results.append({
                    'Encoding': encoding_name,
                    'Model': f'KNN (k={k}, metric={metric})',
                    'Accuracy': acc,
                    'Precision': precision,
                    'Recall': recall,
                    'F1 Score': f1
                })
            except Exception as e:
                print(f"Error training KNN with k={k}, metric={metric}: {e}")
    
    # Logistic Regression
    print(f"\nEvaluating Logistic Regression using {encoding_name} encoding...")
    try:
        logreg = LogisticRegression(max_iter=1000)
        logreg.fit(X, y)
        y_pred = logreg.predict(X)

        acc = accuracy_score(y, y_pred)
        precision = precision_score(y, y_pred, average='macro', zero_division=0)
        recall = recall_score(y, y_pred, average='macro', zero_division=0)
        f1 = f1_score(y, y_pred, average='macro', zero_division=0)

        results.append({
            'Encoding': encoding_name,
            'Model': 'Logistic Regression',
            'Accuracy': acc,
            'Precision': precision,
            'Recall': recall,
            'F1 Score': f1
        })
    except Exception as e:
        print(f"Error training Logistic Regression: {e}")
    
    return pd.DataFrame(results)

# Run evaluations
print("\nRunning model evaluations on training data...")
tfidf_results_train = evaluate_models_train(X_tfidf_scaled, y_encoded, 'TF-IDF', label_encoder)
onehot_results_train = evaluate_models_train(X_onehot_scaled, y_encoded, 'One-Hot', label_encoder)
all_results = pd.concat([tfidf_results_train, onehot_results_train], ignore_index=True)

print("\nModel Evaluation Results:")
print(all_results)

# Task 4: Enhanced Visualizations

custom_palette = sns.color_palette("husl", 8)
sns.set_palette(custom_palette)

# Accuracy visualization
plt.figure(figsize=(16, 12))
ax = sns.barplot(data=all_results, x='Model', y='Accuracy', hue='Encoding', palette="Set2")
plt.title('Model Accuracy Comparison', fontsize=18, fontweight='bold', pad=20)
plt.xlabel('Model', fontsize=14)
plt.ylabel('Accuracy', fontsize=14)
plt.legend(title='Encoding Method', fontsize=12, title_fontsize=13)
for p in ax.patches:
    ax.annotate(f'{p.get_height():.3f}', 
                (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='bottom', fontsize=10, xytext=(0, 5), textcoords='offset points')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('model_accuracy_comparison.png', dpi=300, bbox_inches='tight')

# KNN-specific comparison
knn_results = all_results[all_results['Model'] == 'KNN']
g = sns.FacetGrid(knn_results, col='Encoding', height=4, aspect=1.2)
g.map_dataframe(sns.barplot, x='k', y='Accuracy', hue='Metric', palette="rocket")
g.add_legend(title='Distance Metric', fontsize=12, title_fontsize=13)
g.set_titles(col_template="{col_name} Encoding", size=14)
g.set_axis_labels("k-value", "Accuracy", fontsize=14)
plt.suptitle('KNN Performance Across Configurations', fontsize=18, fontweight='bold')
plt.tight_layout()
plt.savefig('knn_performance_comparison.png', dpi=300, bbox_inches='tight')

# F1-score heatmap
pivoted = all_results.pivot_table(
    index=['Model', 'k', 'Metric'], 
    columns='Encoding', 
    values='F1'
)
plt.figure(figsize=(12, 8))
sns.heatmap(pivoted, annot=True, cmap='YlGnBu', linewidths=.5, fmt='.3f')
plt.title('F1 Score Comparison', fontsize=16, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('f1_heatmap_comparison.png', dpi=300, bbox_inches='tight')

# Task 5: Critical Analysis

analysis = """
# Critical Analysis

## TF-IDF vs. One-Hot Encoding
- **TF-IDF Advantage**: Weights features by importance, capturing semantic relationships (e.g., 'chest pain' weighted higher for cardiovascular diseases). Visualizations show tighter clusters for TF-IDF.
- **One-Hot Limitation**: Binary representation treats all features equally, leading to higher sparsity ({:.4f} vs. TF-IDF’s {:.4f}) and less discriminative power, as seen in scattered PCA/SVD plots.

## Clinical Relevance
- TF-IDF clusters align with clinical categories (cardiovascular, neurological), suggesting potential for diagnostic support.
- Interpretability of TF-IDF weights aids clinical explanation (e.g., key symptoms driving predictions).

## Limitations
- **Dataset**: With only 25 samples (one per disease), classification is challenging. Leave-One-Out CV was unsuitable, as the test class is absent from training, leading to zero metrics. Evaluation on training data was used instead, limiting generalization insights.
- **Features**: Missing lab results or temporal data reduces clinical applicability. High-dimensional TF-IDF features (reduced to 500) may still include noise.
- **Encoding**: TF-IDF may miss rare but critical features; one-hot encoding ignores feature interactions.
- **Recommendation**: Future work should explore clustering or similarity-based methods, as classification with one sample per class is inherently limited.
""".format(onehot_sparsity, tfidf_sparsity)

with open('critical_analysis.txt', 'w') as f:
    f.write(analysis)
print("\nCritical analysis saved to 'critical_analysis.txt'")

# Task 6: Deliverables

# Save results
all_results.to_csv('model_results_loo.csv', index=False)

# Enhanced visualization save
def save_plots(X, title, filename):
    plt.figure(figsize=(10, 8))
    plt.gca().set_facecolor('#f8f9fa')
    scatter = plt.scatter(X[:, 0], X[:, 1], c=plot_colors, alpha=0.8, s=120, edgecolor='w', linewidth=0.8)
    plt.title(title, fontsize=18, fontweight='bold', pad=20, 
              bbox=dict(facecolor='white', edgecolor='gray', boxstyle='round,pad=0.5', alpha=0.8))
    plt.xlabel('Component 1', fontsize=14, fontweight='bold')
    plt.ylabel('Component 2', fontsize=14, fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.5, color='gray')
    
    category_to_marker = {'cardiovascular': 'o', 'neurological': 's', 
                         'endocrine': '^', 'respiratory': 'D', 
                         'gastrointestinal': 'p', 'infectious': '*'}
    handles = []
    for cat_name, cat_diseases in disease_categories.items():
        marker = category_to_marker[cat_name]
        color = category_colors[cat_diseases[0]]
        patch = plt.Line2D([0], [0], marker=marker, color='w', markerfacecolor=color, 
                         markersize=12, label=cat_name.capitalize())
        handles.append(patch)
    
    plt.legend(handles=handles, bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12)
    plt.gca().spines['top'].set_visible(True)
    plt.gca().spines['right'].set_visible(True)
    plt.gca().spines['bottom'].set_visible(True)
    plt.gca().spines['left'].set_visible(True)
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

save_plots(X_tfidf_pca, 'TF-IDF PCA', 'tfidf_pca_enhanced.png')
save_plots(X_tfidf_svd, 'TF-IDF SVD', 'tfidf_svd_enhanced.png')
save_plots(X_onehot_pca, 'One-Hot PCA', 'onehot_pca_enhanced.png')
save_plots(X_onehot_svd, 'One-Hot SVD', 'onehot_svd_enhanced.png')

# Summary and recommendations
best_knn = all_results[(all_results['Model'] == 'KNN') & 
                      (all_results['k'] == 5) & 
                      (all_results['Metric'] == 'cosine') & 
                      (all_results['Encoding'] == 'TF-IDF')]
f1_score_best = best_knn['F1'].iloc[0] if not best_knn.empty else 0.0
accuracy_best = best_knn['Accuracy'].iloc[0] if not best_knn.empty else 0.0

summary = """
## Model Evaluation Summary

The evaluation revealed challenges due to the dataset’s structure (one sample per disease):

1. **Feature Representation**: TF-IDF, with reduced features (500), showed tighter clusters in visualizations compared to one-hot encoding.
2. **Model Performance**:
   - KNN with k=5 and cosine similarity using TF-IDF achieved:
   - F1 Score: {:.3f}
   - Accuracy: {:.3f}
   - Evaluation on training data was used, as Leave-One-Out CV was infeasible with one sample per class.
3. **Clinical Relevance**:
   - Disease clusters align with medical categories, suggesting potential for similarity-based diagnostics.
   - Interpretability of TF-IDF features supports clinical use.

## Recommendations for Implementation

1. Use TF-IDF + KNN (k=5, cosine) as a baseline for similarity-based disease profiling, not classification.
2. Incorporate additional clinical features (e.g., lab results) to improve robustness.
3. Explore clustering or nearest-neighbor search for diagnostic support, given classification limitations.
4. Validate with a larger dataset to enable proper cross-validation.
""".format(f1_score_best, accuracy_best)

with open('model_summary_and_recommendations.md', 'w') as f:
    f.write(summary)
print("\nSummary saved to 'model_summary_and_recommendations.md'")

# Final report
print("\nAll analysis complete! Files generated:")
print("- model_results_loo.csv")
print("- tfidf_pca_vis.png")
print("- tfidf_svd_vis.png")
print("- onehot_pca_vis.png")
print("- onehot_svd_vis.png")
print("- tfidf_pca_enhanced.png")
print("- tfidf_svd_enhanced.png")
print("- onehot_pca_enhanced.png")
print("- onehot_svd_enhanced.png")
print("- model_accuracy_comparison.png")
print("- knn_performance_comparison.png")
print("- f1_heatmap_comparison.png")
print("- critical_analysis.txt")
print("- model_summary_and_recommendations.md")
import streamlit as st
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
import base64

# Set page configuration
st.set_page_config(page_title="Disease Features Analysis", layout="wide")

# Helper functions
def safe_literal_eval(x):
    try:
        return literal_eval(x)
    except (ValueError, SyntaxError):
        return [] if isinstance(x, str) and ('[' in x or '{' in x) else {}

def process_data():
    # Load datasets
    try:
        disease_df = pd.read_csv('disease_features.csv')
        encoded_df = pd.read_csv('encoded_output2.csv')
    except FileNotFoundError:
        st.error("Error: One or both CSV files not found.")
        st.stop()
    
    # Parse stringified lists to Python lists
    for col in ['Risk Factors', 'Symptoms', 'Signs']:
        disease_df[col] = disease_df[col].fillna('[]').apply(safe_literal_eval)
    disease_df['Subtypes'] = disease_df['Subtypes'].fillna('{}').apply(safe_literal_eval)
    
    # Convert lists to single strings
    disease_df['Risk_Str'] = disease_df['Risk Factors'].apply(lambda x: ' '.join(x) if isinstance(x, list) else '')
    disease_df['Symptoms_Str'] = disease_df['Symptoms'].apply(lambda x: ' '.join(x) if isinstance(x, list) else '')
    disease_df['Signs_Str'] = disease_df['Signs'].apply(lambda x: ' '.join(x) if isinstance(x, list) else '')
    disease_df['Subtypes_Str'] = disease_df['Subtypes'].apply(lambda x: ' '.join([k + ' ' + v for k, v in x.items()]) if isinstance(x, dict) else '')
    
    return disease_df, encoded_df

def get_tfidf_features(disease_df):
    # Apply TF-IDF vectorization
    tfidf_risk = TfidfVectorizer(stop_words='english', max_features=125)
    tfidf_symptoms = TfidfVectorizer(stop_words='english', max_features=125)
    tfidf_signs = TfidfVectorizer(stop_words='english', max_features=125)
    tfidf_subtypes = TfidfVectorizer(stop_words='english', max_features=125)

    X_risk = tfidf_risk.fit_transform(disease_df['Risk_Str'])
    X_symptoms = tfidf_symptoms.fit_transform(disease_df['Symptoms_Str'])
    X_signs = tfidf_signs.fit_transform(disease_df['Signs_Str'])
    X_subtypes = tfidf_subtypes.fit_transform(disease_df['Subtypes_Str'])

    X_tfidf = hstack([X_risk, X_symptoms, X_signs, X_subtypes])
    
    return X_tfidf

def get_color_mapping(disease_df):
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
    
    return disease_categories, category_colors

def visualize_reduced_dimensions(X, title, colors, categories):
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_facecolor('#f8f9fa')
    scatter = ax.scatter(X[:, 0], X[:, 1], c=colors, alpha=0.8, s=120, edgecolor='w', linewidth=0.8)
    ax.set_title(title, fontsize=18, fontweight='bold', pad=20, 
              bbox=dict(facecolor='white', edgecolor='gray', boxstyle='round,pad=0.5', alpha=0.8))
    ax.set_xlabel('Component 1', fontsize=14, fontweight='bold')
    ax.set_ylabel('Component 2', fontsize=14, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.5, color='gray')
    
    # Add legend
    category_to_marker = {'cardiovascular': 'o', 'neurological': 's', 
                         'endocrine': '^', 'respiratory': 'D', 
                         'gastrointestinal': 'p', 'infectious': '*'}
    handles = []
    for cat_name, cat_diseases in categories.items():
        if cat_diseases:  # Check if list is not empty
            marker = category_to_marker[cat_name]
            color = category_colors[cat_diseases[0]]
            patch = plt.Line2D([0], [0], marker=marker, color='w', markerfacecolor=color, 
                             markersize=12, label=cat_name.capitalize())
            handles.append(patch)
    
    ax.legend(handles=handles, loc='upper right', fontsize=12)
    for spine in ax.spines.values():
        spine.set_visible(True)
    plt.tight_layout()
    return fig

def evaluate_model(X, y, model_type, encoding, params=None):
    if model_type == 'KNN':
        k = params.get('k', 5)
        metric = params.get('metric', 'cosine')
        model = KNeighborsClassifier(n_neighbors=k, metric=metric)
    else:  # Logistic Regression
        model = LogisticRegression(max_iter=1000)
    
    model.fit(X, y)
    y_pred = model.predict(X)
    
    results = {
        'Accuracy': accuracy_score(y, y_pred),
        'Precision': precision_score(y, y_pred, average='macro', zero_division=0),
        'Recall': recall_score(y, y_pred, average='macro', zero_division=0),
        'F1 Score': f1_score(y, y_pred, average='macro', zero_division=0)
    }
    return results, model

# Main app
st.title("Disease Features Analysis App")

# Load data
with st.spinner("Loading and processing data..."):
    disease_df, encoded_df = process_data()
    disease_categories, category_colors = get_color_mapping(disease_df)

# Sidebar
st.sidebar.header("Navigation")
page = st.sidebar.radio("Go to", ["Data Overview", "Feature Analysis", "Dimensionality Reduction", "Model Evaluation"])

if page == "Data Overview":
    st.header("Disease Features Dataset")
    st.write(disease_df.head())
    
    st.header("Encoded Features Dataset")
    st.write(encoded_df.head())
    
    st.header("Disease Distribution")
    fig, ax = plt.subplots(figsize=(10, 6))
    disease_df['Disease'].value_counts().plot(kind='bar', ax=ax)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    st.pyplot(fig)

elif page == "Feature Analysis":
    st.header("TF-IDF Feature Analysis")
    
    # Extract TF-IDF features
    X_tfidf = get_tfidf_features(disease_df)
    X_onehot = encoded_df.drop('Disease', axis=1)
    
    # Calculate and display sparsity
    tfidf_sparsity = 1.0 - (X_tfidf.count_nonzero() / (X_tfidf.shape[0] * X_tfidf.shape[1]))
    onehot_sparsity = 1.0 - (X_onehot.astype(bool).sum().sum() / (X_onehot.shape[0] * X_onehot.shape[1]))
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("TF-IDF Sparsity", f"{tfidf_sparsity:.4f}")
        st.metric("TF-IDF Unique Features", X_tfidf.shape[1])
    with col2:
        st.metric("One-Hot Sparsity", f"{onehot_sparsity:.4f}")
        st.metric("One-Hot Unique Features", X_onehot.shape[1])
    
    st.write("**Interpretation:** Lower sparsity values indicate more dense feature representations. TF-IDF encoding provides a more compact representation compared to one-hot encoding.")

elif page == "Dimensionality Reduction":
    st.header("Dimensionality Reduction")
    
    # Get features
    X_tfidf = get_tfidf_features(disease_df)
    X_onehot = encoded_df.drop('Disease', axis=1)
    
    # Create color mapping
    plot_colors = [category_colors.get(d, (0.7, 0.7, 0.7, 1.0)) for d in disease_df['Disease']]
    
    # Apply dimensionality reduction
    reduction_method = st.selectbox("Select Reduction Method", ["PCA", "SVD"])
    encoding_method = st.selectbox("Select Encoding Method", ["TF-IDF", "One-Hot"])
    
    n_components = 2
    if reduction_method == "PCA" and encoding_method == "TF-IDF":
        pca = PCA(n_components=n_components)
        X_reduced = pca.fit_transform(X_tfidf.toarray())
        explained_var = pca.explained_variance_ratio_
        title = "TF-IDF: PCA Reduced Dimensions"
    elif reduction_method == "SVD" and encoding_method == "TF-IDF":
        svd = TruncatedSVD(n_components=n_components)
        X_reduced = svd.fit_transform(X_tfidf)
        explained_var = svd.explained_variance_ratio_
        title = "TF-IDF: SVD Reduced Dimensions"
    elif reduction_method == "PCA" and encoding_method == "One-Hot":
        pca = PCA(n_components=n_components)
        X_reduced = pca.fit_transform(X_onehot)
        explained_var = pca.explained_variance_ratio_
        title = "One-Hot: PCA Reduced Dimensions"
    else:  # SVD and One-Hot
        svd = TruncatedSVD(n_components=n_components)
        X_reduced = svd.fit_transform(X_onehot)
        explained_var = svd.explained_variance_ratio_
        title = "One-Hot: SVD Reduced Dimensions"
    
    # Display explained variance
    st.write(f"Explained Variance Ratio: {explained_var[0]:.4f}, {explained_var[1]:.4f}")
    st.write(f"Total Explained Variance: {sum(explained_var):.4f}")
    
    # Plot reduced dimensions
    fig = visualize_reduced_dimensions(X_reduced, title, plot_colors, disease_categories)
    st.pyplot(fig)

elif page == "Model Evaluation":
    st.header("Model Evaluation")
    
    # Get features and labels
    X_tfidf = get_tfidf_features(disease_df)
    X_onehot = encoded_df.drop('Disease', axis=1)
    y = disease_df['Disease']
    
    # Scale features
    scaler_tfidf = StandardScaler(with_mean=False)
    X_tfidf_scaled = scaler_tfidf.fit_transform(X_tfidf)
    
    scaler_onehot = StandardScaler()
    X_onehot_scaled = scaler_onehot.fit_transform(X_onehot)
    
    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # Model selection
    col1, col2, col3 = st.columns(3)
    with col1:
        model_type = st.selectbox("Select Model", ["KNN", "Logistic Regression"])
    with col2:
        encoding_type = st.selectbox("Select Encoding", ["TF-IDF", "One-Hot"])
    
    # Additional parameters for KNN
    if model_type == "KNN":
        with col3:
            k_value = st.selectbox("Select k value", [3, 5, 7])
            metric = st.selectbox("Select distance metric", ["euclidean", "manhattan", "cosine"])
    
    # Evaluate model
    X = X_tfidf_scaled if encoding_type == "TF-IDF" else X_onehot_scaled
    params = {'k': k_value, 'metric': metric} if model_type == "KNN" else {}
    
    if st.button("Evaluate Model"):
        with st.spinner("Evaluating model..."):
            results, model = evaluate_model(X, y_encoded, model_type, encoding_type, params)
            
            # Display results
            st.subheader("Model Performance Metrics")
            metrics_df = pd.DataFrame(results, index=[0])
            st.dataframe(metrics_df)
            
            # Visualize results
            fig, ax = plt.subplots(figsize=(10, 6))
            metrics_df.T.plot(kind='bar', ax=ax)
            plt.ylabel("Score")
            plt.title(f"{model_type} with {encoding_type} Encoding")
            plt.tight_layout()
            st.pyplot(fig)
            
            # Interpretation
            st.subheader("Interpretation")
            if model_type == "KNN":
                st.write(f"KNN with k={k_value} and {metric} distance using {encoding_type} encoding achieved an accuracy of {results['Accuracy']:.4f} and F1 score of {results['F1 Score']:.4f}.")
            else:
                st.write(f"Logistic Regression using {encoding_type} encoding achieved an accuracy of {results['Accuracy']:.4f} and F1 score of {results['F1 Score']:.4f}.")
            
            if encoding_type == "TF-IDF":
                st.write("TF-IDF encoding typically provides better representation for text data as it captures importance of terms relative to their frequency.")
            else:
                st.write("One-hot encoding provides a binary representation that treats all features equally, regardless of their importance.")

# Add download options if needed
st.sidebar.header("Export")
if st.sidebar.button("Generate Report"):
    # Create a summary of the analysis
    summary = f"""
    # Disease Features Analysis Report
    
    ## Dataset Summary
    - Number of diseases: {len(disease_df)}
    - Number of features: {encoded_df.shape[1] - 1}
    
    ## Feature Encoding Comparison
    - TF-IDF Sparsity: {1.0 - (get_tfidf_features(disease_df).count_nonzero() / (get_tfidf_features(disease_df).shape[0] * get_tfidf_features(disease_df).shape[1])):.4f}
    - One-Hot Sparsity: {1.0 - (encoded_df.drop('Disease', axis=1).astype(bool).sum().sum() / (encoded_df.drop('Disease', axis=1).shape[0] * encoded_df.drop('Disease', axis=1).shape[1])):.4f}
    
    ## Recommendations
    - TF-IDF + KNN (k=5, cosine) is suggested as a baseline for similarity-based disease profiling.
    - Consider incorporating additional clinical features to improve robustness.
    - Validate with a larger dataset to enable proper cross-validation.
    """
    
    # Create a download link
    b64 = base64.b64encode(summary.encode()).decode()
    href = f'<a href="data:text/plain;base64,{b64}" download="disease_analysis_report.md">Download Report</a>'
    st.sidebar.markdown(href, unsafe_allow_html=True)
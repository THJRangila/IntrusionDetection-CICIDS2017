"""
Network Intrusion Detection System - Streamlit Demo UI
IT4092: Machine Learning | KDU | 2025
"""
import sys
import os

# Add pylibs to path (TensorFlow installed here due to disk space)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'pylibs'))

import streamlit as st
import pandas as pd
import numpy as np
import json
import joblib
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Network Intrusion Detection System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'models')
FIGURES_DIR = os.path.join(BASE_DIR, 'figures')
DATA_DIR = os.path.join(BASE_DIR, 'data', 'processed')

# ============================================================
# LOAD CACHED RESOURCES
# ============================================================
@st.cache_data
def load_metrics():
    metrics_path = os.path.join(MODELS_DIR, 'model_metrics.json')
    if os.path.exists(metrics_path):
        with open(metrics_path, 'r') as f:
            return json.load(f)
    return None

@st.cache_resource
def load_ml_models():
    models = {}
    model_files = {
        'Decision Tree': 'decision_tree.joblib',
        'Random Forest': 'random_forest.joblib',
        'XGBoost': 'xgboost_model.joblib',
    }
    for name, filename in model_files.items():
        path = os.path.join(MODELS_DIR, filename)
        if os.path.exists(path):
            models[name] = joblib.load(path)
    return models

@st.cache_resource
def load_dl_models():
    models = {}
    try:
        import tensorflow as tf
        for name, filename in [('DNN', 'dnn_model.keras'), ('1D-CNN', 'cnn_model.keras')]:
            path = os.path.join(MODELS_DIR, filename)
            if os.path.exists(path):
                models[name] = tf.keras.models.load_model(path)
    except ImportError:
        st.warning("TensorFlow not available. DL models won't load.")
    return models

@st.cache_resource
def load_artifacts():
    artifacts = {}
    for name, filename in [('scaler', 'scaler.joblib'),
                            ('label_encoder', 'label_encoder.joblib'),
                            ('features', 'selected_features.joblib')]:
        path = os.path.join(MODELS_DIR, filename)
        if os.path.exists(path):
            artifacts[name] = joblib.load(path)
    return artifacts

# ============================================================
# SIDEBAR
# ============================================================
st.sidebar.title("🛡️ Network IDS")
st.sidebar.markdown("---")
st.sidebar.markdown("""
**Course:** IT4092 - Machine Learning
**University:** KDU
**Dataset:** CICIDS2017
**Models:** 5 (DT, RF, XGB, DNN, CNN)
""")

# ============================================================
# MAIN CONTENT - TABS
# ============================================================
st.title("🛡️ Network Intrusion Detection System")
st.markdown("*Using Machine Learning on CICIDS2017 Dataset*")

tab1, tab2, tab3, tab4 = st.tabs(["📊 Dataset Overview", "🔍 EDA", "📈 Model Comparison", "🎯 Live Prediction"])

# ============================================================
# TAB 1: DATASET OVERVIEW
# ============================================================
with tab1:
    st.header("Dataset Overview - CICIDS2017")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Records", "2,830,743")
    col2.metric("Features", "78")
    col3.metric("Attack Classes", "9 (grouped)")
    col4.metric("CSV Files", "8")

    st.markdown("---")

    st.subheader("Class Distribution")
    metrics = load_metrics()
    if metrics and 'class_names' in metrics:
        class_names = metrics['class_names']
        st.info(f"Classes: {', '.join(class_names)}")

    # Dataset description
    st.subheader("Dataset Description")
    dataset_info = pd.DataFrame({
        'Day': ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'],
        'Records': ['529,918', '445,909', '692,703', '458,968', '703,245'],
        'Attack Types': [
            'BENIGN only (baseline)',
            'FTP-Patator, SSH-Patator',
            'DoS Hulk, GoldenEye, slowloris, Slowhttp, Heartbleed',
            'Web Attacks (BruteForce, XSS, SQLi), Infiltration',
            'Bot, DDoS, PortScan'
        ]
    })
    st.dataframe(dataset_info, use_container_width=True, hide_index=True)

    st.subheader("Class Grouping Applied")
    grouping = pd.DataFrame({
        'Grouped Class': ['BENIGN', 'DoS', 'DDoS', 'PortScan', 'Brute Force',
                         'Web Attack', 'Bot', 'Infiltration', 'Heartbleed'],
        'Original Labels': [
            'BENIGN',
            'DoS Hulk, DoS GoldenEye, DoS slowloris, DoS Slowhttptest',
            'DDoS',
            'PortScan',
            'FTP-Patator, SSH-Patator',
            'Web Attack - Brute Force, XSS, SQL Injection',
            'Bot',
            'Infiltration',
            'Heartbleed (11 samples only)'
        ]
    })
    st.dataframe(grouping, use_container_width=True, hide_index=True)

    st.subheader("Preprocessing Pipeline")
    st.markdown("""
    1. **Data Consolidation** - Merged 8 CSV files, stripped column whitespace
    2. **Missing Value Handling** - Replaced infinity → NaN, median imputation
    3. **Duplicate Removal** - Removed duplicate records
    4. **Outlier Clipping** - Clipped at 1st/99th percentile
    5. **Feature Selection** - Zero-variance → High correlation → Mutual Information (78 → ~35 features)
    6. **Feature Scaling** - StandardScaler
    7. **Class Imbalance** - Undersample BENIGN + SMOTE minorities
    """)

# ============================================================
# TAB 2: EDA
# ============================================================
with tab2:
    st.header("Exploratory Data Analysis")

    # List available figures
    available_figures = {
        'Class Distribution': 'class_distribution.png',
        'Missing Values': 'missing_values.png',
        'Correlation Heatmap': 'correlation_heatmap.png',
        'Feature Box Plots (Benign vs Attack)': 'feature_boxplots.png',
        'PCA Scatter Plot': 'pca_scatter.png',
        'Feature Importance (Mutual Information)': 'feature_importance.png',
    }

    selected_fig = st.selectbox("Select Visualization", list(available_figures.keys()))

    fig_path = os.path.join(FIGURES_DIR, available_figures[selected_fig])
    if os.path.exists(fig_path):
        st.image(fig_path, use_container_width=True)
    else:
        st.warning(f"Figure not found: {available_figures[selected_fig]}. Run the notebook first to generate figures.")

# ============================================================
# TAB 3: MODEL COMPARISON
# ============================================================
with tab3:
    st.header("Model Comparison")

    metrics = load_metrics()

    if metrics is None:
        st.error("No metrics found. Please run the notebook first to train models and save metrics.")
    else:
        model_metrics = metrics.get('model_metrics', {})
        model_names = list(model_metrics.keys())

        # Summary metrics table
        st.subheader("Performance Summary")
        summary_data = []
        for name in model_names:
            m = model_metrics[name]
            summary_data.append({
                'Model': name,
                'Accuracy': f"{m.get('accuracy', 0):.4f}",
                'Precision (Macro)': f"{m.get('precision_macro', 0):.4f}",
                'Recall (Macro)': f"{m.get('recall_macro', 0):.4f}",
                'F1-Score (Macro)': f"{m.get('f1_macro', 0):.4f}",
                'ROC-AUC': f"{m.get('roc_auc', 'N/A')}" if m.get('roc_auc') else 'N/A',
                'Train Time': f"{m.get('train_time', 0):.1f}s"
            })

        summary_df = pd.DataFrame(summary_data)
        st.dataframe(summary_df, use_container_width=True, hide_index=True)

        best_model = metrics.get('best_model', 'Unknown')
        st.success(f"🏆 Best Model (by Macro F1-Score): **{best_model}**")

        st.markdown("---")

        # Interactive bar chart
        st.subheader("Metrics Comparison")
        metric_options = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']

        fig = go.Figure()
        colors = ['#2196F3', '#4CAF50', '#FF9800', '#9C27B0', '#F44336']

        for i, name in enumerate(model_names):
            values = [model_metrics[name].get(m, 0) or 0 for m in metric_options]
            fig.add_trace(go.Bar(
                name=name,
                x=['Accuracy', 'Precision', 'Recall', 'F1-Score'],
                y=values,
                marker_color=colors[i % len(colors)],
                text=[f'{v:.3f}' for v in values],
                textposition='outside'
            ))

        fig.update_layout(
            barmode='group',
            title='Model Performance Comparison',
            yaxis_title='Score',
            yaxis_range=[0, 1.15],
            height=500,
            legend=dict(orientation='h', yanchor='bottom', y=1.02)
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")

        # Confusion Matrix Viewer
        st.subheader("Confusion Matrix")
        selected_model = st.selectbox("Select Model", model_names, key='cm_model')

        cm_data = metrics.get('confusion_matrices', {}).get(selected_model)
        class_names = metrics.get('class_names', [])

        if cm_data and class_names:
            cm = np.array(cm_data)
            # Normalize
            cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            cm_normalized = np.nan_to_num(cm_normalized)

            fig_cm = px.imshow(
                cm_normalized,
                labels=dict(x="Predicted", y="True", color="Recall"),
                x=class_names,
                y=class_names,
                color_continuous_scale='Blues',
                text_auto='.2f',
                title=f'{selected_model} - Normalized Confusion Matrix'
            )
            fig_cm.update_layout(height=600)
            st.plotly_chart(fig_cm, use_container_width=True)

        st.markdown("---")

        # Per-class F1 heatmap
        st.subheader("Per-Class F1-Score")
        class_reports = metrics.get('class_reports', {})

        if class_reports and class_names:
            f1_data = {}
            for name in model_names:
                report = class_reports.get(name, {})
                f1_data[name] = [report.get(cls, {}).get('f1-score', 0) for cls in class_names]

            f1_df = pd.DataFrame(f1_data, index=class_names)

            fig_f1 = px.imshow(
                f1_df.values,
                labels=dict(x="Model", y="Class", color="F1-Score"),
                x=list(f1_data.keys()),
                y=class_names,
                color_continuous_scale='YlOrRd',
                text_auto='.3f',
                title='Per-Class F1-Score Heatmap'
            )
            fig_f1.update_layout(height=500)
            st.plotly_chart(fig_f1, use_container_width=True)

        # Training history images
        st.markdown("---")
        st.subheader("Training History (Deep Learning)")
        col1, col2 = st.columns(2)
        dnn_hist_path = os.path.join(FIGURES_DIR, 'training_history_dnn.png')
        cnn_hist_path = os.path.join(FIGURES_DIR, 'training_history_cnn.png')

        with col1:
            if os.path.exists(dnn_hist_path):
                st.image(dnn_hist_path, caption='DNN Training History', use_container_width=True)
            else:
                st.info("DNN training history not available yet.")
        with col2:
            if os.path.exists(cnn_hist_path):
                st.image(cnn_hist_path, caption='1D-CNN Training History', use_container_width=True)
            else:
                st.info("CNN training history not available yet.")

# ============================================================
# TAB 4: LIVE PREDICTION
# ============================================================
with tab4:
    st.header("🎯 Live Network Traffic Prediction")
    st.markdown("Adjust the feature values below and click **Predict** to classify the network traffic.")

    artifacts = load_artifacts()
    ml_models = load_ml_models()
    dl_models = load_dl_models()

    if not artifacts.get('features') or not artifacts.get('scaler') or not artifacts.get('label_encoder'):
        st.error("Model artifacts not found. Please run the notebook first.")
    elif not ml_models and not dl_models:
        st.error("No trained models found. Please run the notebook first.")
    else:
        features = artifacts['features']
        scaler = artifacts['scaler']
        le = artifacts['label_encoder']

        # Create input sliders for top 10 features
        st.subheader("Input Features")
        n_input_features = min(10, len(features))

        cols = st.columns(2)
        input_values = {}

        for i, feat in enumerate(features[:n_input_features]):
            with cols[i % 2]:
                input_values[feat] = st.number_input(
                    feat,
                    value=0.0,
                    format="%.4f",
                    key=f"feat_{i}"
                )

        # Fill remaining features with 0
        for feat in features[n_input_features:]:
            input_values[feat] = 0.0

        st.markdown("---")

        # Model selection
        all_models = {**ml_models, **dl_models}
        selected_pred_model = st.selectbox("Select Model for Prediction", list(all_models.keys()))

        if st.button("🔍 Predict", type="primary", use_container_width=True):
            # Prepare input
            input_array = np.array([[input_values[f] for f in features]])
            input_scaled = scaler.transform(input_array)

            st.markdown("---")
            st.subheader("Prediction Results")

            # Predict with selected model
            model = all_models[selected_pred_model]

            if selected_pred_model in ['DNN', '1D-CNN']:
                input_for_model = input_scaled.astype(np.float32)
                if selected_pred_model == '1D-CNN':
                    input_for_model = input_for_model.reshape(-1, len(features), 1)
                probs = model.predict(input_for_model, verbose=0)[0]
                pred_class = np.argmax(probs)
            else:
                pred_class = model.predict(input_scaled)[0]
                probs = model.predict_proba(input_scaled)[0]

            predicted_label = le.classes_[pred_class]
            confidence = probs[pred_class] * 100

            # Display result
            if predicted_label == 'BENIGN':
                st.success(f"### ✅ {predicted_label}")
                st.markdown(f"**Confidence:** {confidence:.1f}%")
            else:
                st.error(f"### ⚠️ ATTACK DETECTED: {predicted_label}")
                st.markdown(f"**Confidence:** {confidence:.1f}%")

            # Show probability distribution
            prob_df = pd.DataFrame({
                'Class': le.classes_,
                'Probability': probs * 100
            }).sort_values('Probability', ascending=True)

            fig_prob = px.bar(
                prob_df, x='Probability', y='Class',
                orientation='h',
                title=f'{selected_pred_model} - Class Probabilities',
                color='Probability',
                color_continuous_scale='RdYlGn_r'
            )
            fig_prob.update_layout(height=400, xaxis_title='Probability (%)')
            st.plotly_chart(fig_prob, use_container_width=True)

            # Compare across all models
            st.subheader("All Models Comparison")
            comparison_data = []
            for name, model in all_models.items():
                try:
                    if name in ['DNN', '1D-CNN']:
                        inp = input_scaled.astype(np.float32)
                        if name == '1D-CNN':
                            inp = inp.reshape(-1, len(features), 1)
                        p = model.predict(inp, verbose=0)[0]
                        pred = np.argmax(p)
                        conf = p[pred] * 100
                    else:
                        pred = model.predict(input_scaled)[0]
                        p = model.predict_proba(input_scaled)[0]
                        conf = p[pred] * 100

                    comparison_data.append({
                        'Model': name,
                        'Prediction': le.classes_[pred],
                        'Confidence': f'{conf:.1f}%'
                    })
                except Exception as e:
                    comparison_data.append({
                        'Model': name,
                        'Prediction': f'Error: {str(e)[:50]}',
                        'Confidence': 'N/A'
                    })

            comp_df = pd.DataFrame(comparison_data)
            st.dataframe(comp_df, use_container_width=True, hide_index=True)

# ============================================================
# FOOTER
# ============================================================
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray; font-size: 0.9em;'>
Network Intrusion Detection System | IT4092: Machine Learning |
General Sir John Kotelawala Defence University | 2025
</div>
""", unsafe_allow_html=True)

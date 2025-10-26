import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow import keras
from PIL import Image
import pickle
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from keras.layers import GlobalAveragePooling2D
from keras.models import Model
import pandas as pd
from datetime import datetime
import os

# Page configuration
st.set_page_config(
    page_title="FIRES - Wildfire Detection System",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional CSS styling
st.markdown("""
<style>
    .main {
        padding: 2rem;
    }

    .main-title {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f1f1f;
        margin-bottom: 0.5rem;
        letter-spacing: -0.02em;
    }

    .subtitle {
        font-size: 1.1rem;
        color: #666;
        margin-bottom: 2rem;
        font-weight: 400;
    }

    .fire-detected {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 2rem;
        border-radius: 8px;
        margin: 1rem 0;
        border-left: 4px solid #dc2626;
    }

    .no-fire {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        color: white;
        padding: 2rem;
        border-radius: 8px;
        margin: 1rem 0;
        border-left: 4px solid #059669;
    }

    .stat-box {
        background: #f8fafc;
        padding: 1.5rem;
        border-radius: 6px;
        border: 1px solid #e2e8f0;
        text-align: center;
    }

    .stat-number {
        font-size: 2rem;
        font-weight: 700;
        color: #1e293b;
        margin-bottom: 0.5rem;
    }

    .stat-label {
        font-size: 0.875rem;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    .recommendation-item {
        background: #fffbeb;
        border-left: 3px solid #f59e0b;
        padding: 1rem 1.5rem;
        margin: 0.75rem 0;
        border-radius: 4px;
        color: #1f1f1f;
    }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

IMAGE_ADDRESS="https://insideclimatenews.org/wp-content/uploads/2023/03/wildfire_thibaud-mortiz-afp-getty-2048x1365.jpg"
IMAGE_SIZE=(224, 224)
IMAGE_NAME="user_image.png"
CLASS_LABEL=["nowildfire", "wildfire"]
CLASS_LABEL.sort()
HISTORY_FILE = "fire_detection_history.csv"

@st.cache_resource
def get_MobileNetV2_model():
    base_model = keras.applications.MobileNetV2(include_top=False, weights="imagenet", input_shape=(224, 224, 3), classes=1000, classifier_activation="softmax")
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    model_frozen = Model(inputs=base_model.input, outputs=x)
    return model_frozen

@st.cache_resource
def load_sklearn_models(model_path):
    with open(model_path, "rb") as model_file:
        final_model = pickle.load(model_file)
    return final_model

def featurization(image_path, model):
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=IMAGE_SIZE)
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_batch = np.expand_dims(img_array, axis=0)
    img_preprocessed = preprocess_input(img_batch)
    predictions = model.predict(img_preprocessed)
    return predictions

def log_prediction(result, image_name):
    """Log prediction to CSV file"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    new_record = {
        'timestamp': timestamp,
        'prediction': result,
        'image_name': image_name
    }

    if os.path.exists(HISTORY_FILE):
        df = pd.read_csv(HISTORY_FILE)
        df = pd.concat([df, pd.DataFrame([new_record])], ignore_index=True)
    else:
        df = pd.DataFrame([new_record])

    df.to_csv(HISTORY_FILE, index=False)
    return df

def get_recommendations(prediction):
    """Get actionable recommendations based on prediction"""
    if prediction == "wildfire":
        return [
            "Contact local fire authorities and emergency services immediately",
            "Alert nearby communities and initiate evacuation procedures if necessary",
            "Deploy additional satellite monitoring and thermal imaging",
            "Coordinate with fire suppression resources and ground teams",
            "Monitor wind patterns and weather conditions for fire spread prediction"
        ]
    else:
        return [
            "Continue routine monitoring of the area",
            "Maintain regular satellite imagery analysis schedule",
            "Keep fire prevention protocols active in surrounding regions",
            "Document conditions for historical baseline data"
        ]

# Load models
MobileNetV2_featurize_model = get_MobileNetV2_model()
classification_model = load_sklearn_models("best_ml_model")

def run_app():
    # Sidebar navigation
    with st.sidebar:
        st.markdown("### FIRES")
        st.caption("Wildfire Detection System")
        st.markdown("---")

        page = st.radio(
            "Navigation",
            ["Detection", "Analytics", "About"],
            label_visibility="collapsed"
        )

        st.markdown("---")
        st.caption("97% Detection Accuracy")
        st.caption("MobileNetV2 + MLP Architecture")

    if page == "Detection":
        show_detection_page()
    elif page == "Analytics":
        show_analytics_page()
    else:
        show_about_page()

def show_detection_page():
    st.markdown('<p class="main-title">FIRES</p>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Forest Ignition and Risk Evaluation System</p>', unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("""
    **FIRES** is an advanced AI-powered wildfire detection system that uses satellite imagery analysis
    to identify potential wildfires with 97% accuracy. Upload satellite imagery to get immediate analysis
    and risk assessment.
    """)

    st.markdown("---")

    # Display example wildfire image
    st.image(IMAGE_ADDRESS, caption="Example: Wildfire satellite imagery", use_container_width=True)

    st.markdown("---")

    # File uploader
    uploaded_file = st.file_uploader(
        "Upload satellite image for analysis",
        type=["jpg", "png", "jpeg"],
        help="Supported formats: JPG, PNG, JPEG"
    )

    if uploaded_file:
        col1, col2 = st.columns([1, 1])

        with col1:
            st.markdown("#### Input Image")
            user_image = Image.open(uploaded_file)
            st.image(user_image, use_container_width=True)
            user_image.save(IMAGE_NAME)

            st.caption(f"Filename: {uploaded_file.name}")
            st.caption(f"Size: {user_image.size[0]} x {user_image.size[1]} pixels")

        with col2:
            st.markdown("#### Analysis Results")

            with st.spinner("Processing image..."):
                # Run prediction
                image_features = featurization(IMAGE_NAME, MobileNetV2_featurize_model)
                model_predict = classification_model.predict(image_features)
                result_label = CLASS_LABEL[model_predict[0]]

                # Log prediction
                log_prediction(result_label, uploaded_file.name)

                # Display result
                if result_label == "wildfire":
                    st.markdown(f"""
                    <div class="fire-detected">
                        <h3 style="margin:0; font-weight:600;">WILDFIRE DETECTED</h3>
                        <p style="margin-top:0.5rem; opacity:0.9;">Immediate action required</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="no-fire">
                        <h3 style="margin:0; font-weight:600;">NO WILDFIRE DETECTED</h3>
                        <p style="margin-top:0.5rem; opacity:0.9;">Area appears clear</p>
                    </div>
                    """, unsafe_allow_html=True)

                st.markdown("##### Analysis Details")
                st.text(f"Model: MobileNetV2 + MLP Classifier")
                st.text(f"Accuracy: 97%")

        # Recommendations
        st.markdown("---")
        st.markdown("### Recommended Actions")

        recommendations = get_recommendations(result_label)
        for i, rec in enumerate(recommendations, 1):
            st.markdown(f"""
            <div class="recommendation-item">
                <strong>{i}.</strong> {rec}
            </div>
            """, unsafe_allow_html=True)

def show_analytics_page():
    st.markdown('<p class="main-title">Analytics Dashboard</p>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Historical detection data and statistics</p>', unsafe_allow_html=True)

    st.markdown("---")

    if not os.path.exists(HISTORY_FILE):
        st.info("No analysis history available. Upload images on the Detection page to generate analytics.")
        return

    df = pd.read_csv(HISTORY_FILE)

    # Summary statistics
    st.markdown("### Summary Statistics")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(f"""
        <div class="stat-box">
            <div class="stat-number">{len(df)}</div>
            <div class="stat-label">Total Analyses</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        fire_count = len(df[df['prediction'] == 'wildfire'])
        st.markdown(f"""
        <div class="stat-box">
            <div class="stat-number" style="color:#dc2626;">{fire_count}</div>
            <div class="stat-label">Fires Detected</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        no_fire_count = len(df[df['prediction'] == 'nowildfire'])
        st.markdown(f"""
        <div class="stat-box">
            <div class="stat-number" style="color:#059669;">{no_fire_count}</div>
            <div class="stat-label">Clear Areas</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        fire_rate = (fire_count / len(df) * 100) if len(df) > 0 else 0
        st.markdown(f"""
        <div class="stat-box">
            <div class="stat-number">{fire_rate:.1f}%</div>
            <div class="stat-label">Detection Rate</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Recent history
    st.markdown("### Recent Analyses")
    recent_df = df.sort_values('timestamp', ascending=False).head(20)
    st.dataframe(recent_df, use_container_width=True, hide_index=True)

    # Export data
    st.markdown("---")
    st.markdown("### Export Data")

    csv = df.to_csv(index=False)
    st.download_button(
        label="Download Full History (CSV)",
        data=csv,
        file_name=f"fire_detection_history_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv"
    )

def show_about_page():
    st.markdown('<p class="main-title">About FIRES</p>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Technical specifications and project information</p>', unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("""
    ### Overview

    FIRES (Forest Ignition and Risk Evaluation System) is an advanced machine learning system
    designed to detect wildfires from satellite imagery with high accuracy. The system leverages transfer
    learning and deep neural networks to provide rapid, automated fire detection capabilities.

    ### Technical Architecture

    **Feature Extraction Layer**
    - Model: MobileNetV2 (pre-trained on ImageNet)
    - Input Resolution: 224x224 RGB
    - Feature Dimension: 1280-dimensional vectors
    - Pooling: Global Average Pooling

    **Classification Layer**
    - Algorithm: Multi-Layer Perceptron (MLP)
    - Hidden Layer: 100 neurons
    - Activation: ReLU
    - Optimizer: Adam
    - Output: Binary classification (fire/no-fire)

    ### Model Performance

    - **Accuracy:** 97%
    - **Training:** Custom satellite imagery dataset
    - **Validation:** Cross-validated on diverse geographical regions

    ### Use Cases

    - Early wildfire detection for rapid response
    - Continuous monitoring of fire-prone regions
    - Historical fire pattern analysis
    - Risk assessment for forestry management

    ### Technology Stack

    - **Frontend:** Streamlit
    - **Deep Learning:** TensorFlow, Keras
    - **Machine Learning:** Scikit-learn
    - **Data Processing:** NumPy, Pandas
    - **Image Processing:** PIL

    ### Congressional App Challenge 2025

    This application was developed for the Congressional App Challenge to demonstrate how
    artificial intelligence and satellite technology can be leveraged to protect communities
    and natural resources from the devastating effects of wildfires.

    ---

    © 2025 FIRES Project | Developed for environmental protection and community safety
    """)

run_app()

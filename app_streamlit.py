"""
Scalp Hair Disease Detection System - ENHANCED Streamlit Application
Author: Your Name
Date: October 22, 2025
Framework: Streamlit + YOLOv11
Features: Detection + Heatmaps + Confidence Analysis + Training Metrics
"""

import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import io

# ==================== PAGE CONFIGURATION ====================
st.set_page_config(
    page_title="Scalp Hair Disease Detection",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== CUSTOM CSS STYLING ====================
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 700;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .confidence-high {
        color: #4CAF50;
        font-weight: bold;
        font-size: 1.2em;
    }
    .confidence-medium {
        color: #FF9800;
        font-weight: bold;
        font-size: 1.2em;
    }
    .confidence-low {
        color: #F44336;
        font-weight: bold;
        font-size: 1.2em;
    }
</style>
""", unsafe_allow_html=True)

# ==================== MODEL LOADING ====================
@st.cache_resource
def load_model():
    """Load YOLOv11 model with caching"""
    try:
        model = YOLO("runs/detects/train7/weights/best.pt")
        return model
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None

# ==================== DISEASE INFORMATION ====================
DISEASE_INFO = {
    'Alopecia Areata': {
        'description': 'An autoimmune disorder where the immune system attacks hair follicles.',
        'symptoms': 'Circular bald patches, nail changes, sudden hair loss',
        'treatment': 'Corticosteroid injections, topical immunotherapy, JAK inhibitors',
        'severity': 'Moderate to Severe',
        'specialist': 'Dermatologist, Immunologist'
    },
    'Contact Dermatitis': {
        'description': 'Skin inflammation from direct contact with irritants or allergens.',
        'symptoms': 'Redness, itching, scaling, burning sensation',
        'treatment': 'Avoid irritants, topical corticosteroids, antihistamines',
        'severity': 'Mild to Moderate',
        'specialist': 'Dermatologist, Allergist'
    },
    'Folliculitis': {
        'description': 'Inflammation or infection of hair follicles.',
        'symptoms': 'Small red bumps, white pustules, itching, tenderness',
        'treatment': 'Antibacterial shampoos, topical/oral antibiotics',
        'severity': 'Mild to Moderate',
        'specialist': 'Dermatologist'
    },
    'Head_Lice': {
        'description': 'Parasitic infestation of the scalp.',
        'symptoms': 'Intense itching, visible nits, scalp irritation',
        'treatment': 'Medicated shampoos, manual removal, oral ivermectin',
        'severity': 'Mild',
        'specialist': 'Primary Care Physician'
    },
    'Psoriasis': {
        'description': 'Chronic autoimmune skin condition.',
        'symptoms': 'Silver-white scales, red patches, itching',
        'treatment': 'Coal tar shampoo, salicylic acid, biologics',
        'severity': 'Moderate to Severe',
        'specialist': 'Dermatologist, Rheumatologist'
    },
    'dandruff': {
        'description': 'Common scalp condition with excessive flaking.',
        'symptoms': 'White/yellow flakes, itchy scalp',
        'treatment': 'Anti-dandruff shampoos with zinc pyrithione',
        'severity': 'Mild',
        'specialist': 'Dermatologist'
    },
    'dry hair': {
        'description': 'Lack of moisture in hair strands.',
        'symptoms': 'Brittle, dull hair, split ends, frizz',
        'treatment': 'Deep conditioning, coconut oil, reduce heat',
        'severity': 'Mild',
        'specialist': 'Trichologist'
    },
    'grey hair': {
        'description': 'Natural aging process of pigmentation loss.',
        'symptoms': 'Gray or white hair strands',
        'treatment': 'Hair dye (cosmetic) or embrace natural look',
        'severity': 'Normal Aging',
        'specialist': 'None required'
    },
    'low hair density': {
        'description': 'Reduced number of hair follicles.',
        'symptoms': 'Thin-looking hair, visible scalp',
        'treatment': 'Minoxidil, finasteride, PRP therapy',
        'severity': 'Mild to Moderate',
        'specialist': 'Trichologist, Hair Transplant Surgeon'
    }
}

# ==================== UTILITY FUNCTIONS ====================

def generate_heatmap(image, boxes):
    """Generate heatmap from detection boxes"""
    img_array = np.array(image)
    heatmap = np.zeros((img_array.shape[0], img_array.shape[1]), dtype=np.float32)
    
    for box in boxes:
        x1, y1, x2, y2 = map(int, box)
        heatmap[y1:y2, x1:x2] += 1
    
    # Normalize and colorize
    heatmap_norm = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX)
    heatmap_color = cv2.applyColorMap(heatmap_norm.astype(np.uint8), cv2.COLORMAP_JET)
    
    # Overlay
    overlay = cv2.addWeighted(img_array, 0.6, heatmap_color, 0.4, 0)
    return Image.fromarray(overlay)

def plot_confidence_distribution(detections):
    """Plot confidence score distribution"""
    if not detections:
        return None
    
    confidences = [det['confidence'] for det in detections]
    
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(confidences, bins=10, color='skyblue', edgecolor='black', alpha=0.7)
    ax.set_xlabel('Confidence Score', fontsize=12)
    ax.set_ylabel('Number of Detections', fontsize=12)
    ax.set_title('Confidence Score Distribution', fontsize=14, fontweight='bold')
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    return fig

def load_training_metrics():
    """Load training metrics from results.csv"""
    try:
        csv_path = "runs/detects/train7/results.csv"
        df = pd.read_csv(csv_path)
        return df
    except Exception as e:
        st.error(f"Could not load training metrics: {e}")
        return None

def plot_training_metrics(df):
    """Plot training performance metrics"""
    if df is None:
        return None
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    epochs = df["epoch"] if "epoch" in df.columns else range(len(df))
    precision = df["metrics/precision(B)"] if "metrics/precision(B)" in df.columns else None
    recall = df["metrics/recall(B)"] if "metrics/recall(B)" in df.columns else None
    map50 = df["metrics/mAP50(B)"] if "metrics/mAP50(B)" in df.columns else None
    
    if precision is not None:
        ax.plot(epochs, precision, label='Precision', marker='o', linewidth=2)
    if recall is not None:
        ax.plot(epochs, recall, label='Recall', marker='s', linewidth=2)
    if map50 is not None:
        ax.plot(epochs, map50, label='mAP@0.5', marker='^', linewidth=2)
    
    ax.set_xlabel('Epochs', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Model Performance Metrics (Train7)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    return fig

# ==================== PREDICTION FUNCTION ====================
def predict_image(image, conf_threshold):
    """Run YOLO prediction with all visualizations"""
    model = load_model()
    if model is None:
        return None, None, None, None
    
    try:
        img_array = np.array(image)
        
        # Run prediction
        results = model.predict(
            source=img_array,
            conf=conf_threshold,
            save=False,
            verbose=False
        )
        
        # Get annotated image
        annotated_img = results[0].plot()
        annotated_img = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
        
        # Extract detections
        detections = []
        boxes = []
        for box in results[0].boxes:
            cls_id = int(box.cls.item())
            conf = box.conf.item()
            label = model.names[cls_id]
            xyxy = box.xyxy[0].cpu().numpy()
            
            detections.append({
                'disease': label,
                'confidence': conf
            })
            boxes.append(xyxy)
        
        # Generate heatmap
        heatmap_img = generate_heatmap(image, boxes) if boxes else None
        
        # Generate confidence plot
        conf_plot = plot_confidence_distribution(detections)
        
        return Image.fromarray(annotated_img), detections, heatmap_img, conf_plot
    
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        return None, None, None, None

# ==================== MAIN APPLICATION ====================
def main():
    # Header
    st.markdown('<h1 class="main-header">🔬 Scalp Hair Disease Detection System</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">AI-Powered Diagnosis using YOLOv11 Deep Learning Model</p>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("---")
        st.header("⚙️ Settings")
        
        conf_threshold = st.slider(
            "Confidence Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.25,
            step=0.05,
            help="Lower values detect more objects"
        )
        
        st.markdown("---")
        st.header("ℹ️ About")
        st.info("""
            **Dataset:** 888 images from Roboflow
            **Model:** YOLOv11 (100+ epochs)
            **Classes:** 9 disease types
            **Precision:** ~60%
            **Recall:** ~35%
        """)
        
        st.markdown("---")
        st.header("⚠️ Medical Disclaimer")
        st.warning("This is for educational purposes only. Not for medical diagnosis.")
    
    # Main Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📸 Detection & Analysis",
        "📊 Training Metrics",
        "📚 Disease Information",
        "❓ How to Use"
    ])
    
    # ==================== TAB 1: DETECTION ====================
    with tab1:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("📤 Upload Scalp/Hair Image")
            
            uploaded_file = st.file_uploader(
                "Choose an image",
                type=['jpg', 'jpeg', 'png', 'bmp']
            )
            
            if uploaded_file:
                image = Image.open(uploaded_file)
                st.image(image, caption="Uploaded Image", use_column_width=True)
                
                if st.button("🔍 Analyze Image", type="primary"):
                    with st.spinner("🔄 Analyzing..."):
                        annotated, detections, heatmap, conf_plot = predict_image(image, conf_threshold)
                        
                        if annotated:
                            st.session_state['annotated'] = annotated
                            st.session_state['detections'] = detections
                            st.session_state['heatmap'] = heatmap
                            st.session_state['conf_plot'] = conf_plot
                            st.success("✅ Analysis complete!")
        
        with col2:
            st.subheader("📊 Results")
            
            if 'annotated' in st.session_state:
                # Show annotated image
                st.image(st.session_state['annotated'], caption="Detection Results", use_column_width=True)
                
                num_det = len(st.session_state['detections'])
                st.success(f"✅ Found **{num_det}** detection(s)")
        
        # Show additional visualizations
        if 'heatmap' in st.session_state and st.session_state['heatmap']:
            st.markdown("---")
            st.subheader("🔥 Heatmap Visualization")
            
            col_a, col_b = st.columns([1, 1])
            with col_a:
                st.image(st.session_state['heatmap'], caption="Detection Heatmap", use_column_width=True)
            with col_b:
                st.info("""
                **What is a Heatmap?**
                
                The heatmap shows areas of high detection concentration:
                - 🔴 **Red zones** = High detection activity
                - 🟡 **Yellow zones** = Moderate activity  
                - 🔵 **Blue zones** = Low/no activity
                
                This helps visualize problem areas on the scalp.
                """)
        
        # Show confidence distribution
        if 'conf_plot' in st.session_state and st.session_state['conf_plot']:
            st.markdown("---")
            st.subheader("📈 Confidence Score Analysis")
            st.pyplot(st.session_state['conf_plot'])
        
        # Show detection details
        if 'detections' in st.session_state and st.session_state['detections']:
            st.markdown("---")
            st.subheader("🔬 Detected Conditions")
            
            for i, det in enumerate(st.session_state['detections'], 1):
                disease = det['disease']
                confidence = det['confidence']
                
                if confidence >= 0.7:
                    conf_emoji = "🟢"
                    conf_class = "confidence-high"
                elif confidence >= 0.4:
                    conf_emoji = "🟡"
                    conf_class = "confidence-medium"
                else:
                    conf_emoji = "🔴"
                    conf_class = "confidence-low"
                
                with st.expander(f"{conf_emoji} Detection #{i}: **{disease}**", expanded=True):
                    st.markdown(f"**Confidence:** <span class='{conf_class}'>{confidence:.1%}</span>", unsafe_allow_html=True)
                    
                    if disease in DISEASE_INFO:
                        info = DISEASE_INFO[disease]
                        st.markdown(f"**📝 Description:** {info['description']}")
                        st.markdown(f"**🩺 Symptoms:** {info['symptoms']}")
                        st.markdown(f"**💊 Treatment:** {info['treatment']}")
                        st.markdown(f"**⚠️ Severity:** {info['severity']}")
                        st.markdown(f"**👨‍⚕️ Specialist:** {info['specialist']}")
    
   # ==================== TAB 2: TRAINING METRICS ====================
    with tab2:
        st.subheader("📊 Model Training Performance")
        
        # Try to load training metrics CSV
        csv_path = "runs/detects/train7/results.csv"
        
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            fig = plot_training_metrics(df)
            if fig:
                st.pyplot(fig)
            
            st.markdown("---")
            st.subheader("📋 Detailed Training Metrics")
            st.dataframe(df.tail(10), use_container_width=True)
            
            # Summary statistics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                if "metrics/precision(B)" in df.columns:
                    st.metric("Final Precision", f"{df['metrics/precision(B)'].iloc[-1]:.2%}")
            with col2:
                if "metrics/recall(B)" in df.columns:
                    st.metric("Final Recall", f"{df['metrics/recall(B)'].iloc[-1]:.2%}")
            with col3:
                if "metrics/mAP50(B)" in df.columns:
                    st.metric("Final mAP@0.5", f"{df['metrics/mAP50(B)'].iloc[-1]:.2%}")
            with col4:
                st.metric("Total Epochs", len(df))
        
        # If CSV not found, show accuracy graph image
        elif os.path.exists("accuracy_graph.jpg"):
            st.image("accuracy_graph.jpg", caption="Training Performance Metrics (Train7)", use_column_width=True)
            
            st.markdown("---")
            st.subheader("📊 Performance Summary")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Model Precision", "~60%", help="Based on training results")
            with col2:
                st.metric("Model Recall", "~35%", help="Based on training results")
            with col3:
                st.metric("mAP@0.5", "~35%", help="Based on training results")
            
            st.info("""
            **Training Information:**
            - **Total Epochs:** 100+
            - **Dataset:** 888 images from Roboflow
            - **Architecture:** YOLOv11
            - **Training Time:** ~8 hours (estimated)
            
            The graph above shows model performance across all training epochs.
            """)
        
        else:
            st.warning("""
            **Training metrics not found.**
            
            Expected locations:
            - `runs/detect/train7/results.csv` (metrics data)
            - `accuracy_graph.jpg` (performance graph)
            
            Please ensure these files are in your project directory.
            """)
            
            # Show sample metrics
            st.subheader("📊 Model Performance (From Documentation)")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Precision", "~60%")
            with col2:
                st.metric("Recall", "~35%")
            with col3:
                st.metric("mAP@0.5", "~35%")

    
    # ==================== TAB 3: DISEASE INFO ====================
    with tab3:
        st.subheader("📚 Disease Information Database")
        
        for disease, info in DISEASE_INFO.items():
            with st.expander(f"🔬 **{disease}**"):
                col_a, col_b = st.columns([1, 1])
                
                with col_a:
                    st.markdown(f"**📝 Description:**")
                    st.write(info['description'])
                    st.markdown(f"**🩺 Symptoms:**")
                    st.write(info['symptoms'])
                
                with col_b:
                    st.markdown(f"**💊 Treatment:**")
                    st.write(info['treatment'])
                    st.markdown(f"**⚠️ Severity:**")
                    st.write(info['severity'])
                    st.markdown(f"**👨‍⚕️ Specialist:**")
                    st.write(info['specialist'])
    
    # ==================== TAB 4: HOW TO USE ====================
    with tab4:
        st.markdown("""
        ### 📝 Step-by-Step Guide
        
        #### 1️⃣ Upload Image
        - Take a clear, well-lit photo of the scalp
        - Supported formats: JPG, PNG, BMP
        - Recommended: 640x640 pixels or larger
        
        #### 2️⃣ Adjust Confidence (Optional)
        - **0.1-0.3:** More detections (may include false positives)
        - **0.3-0.5:** Balanced (recommended)
        - **0.5-0.9:** High confidence only
        
        #### 3️⃣ Analyze
        - Click "🔍 Analyze Image"
        - Wait 2-5 seconds for processing
        
        #### 4️⃣ View Results
        - **Detection Image:** Bounding boxes around detected areas
        - **Heatmap:** Visual concentration of disease areas
        - **Confidence Plot:** Distribution of detection confidence
        - **Details:** Click each detection for medical info
        
        ### 💡 Features
        
        - ✅ **Real-time Detection** with bounding boxes
        - ✅ **Heatmap Visualization** showing problem areas
        - ✅ **Confidence Analysis** for reliability assessment
        - ✅ **Training Metrics** showing model performance
        - ✅ **Medical Database** with treatment information
        
        ### ⚠️ Important
        
        This system is for **educational purposes only**. Always consult healthcare professionals for medical diagnosis and treatment.
        """)

if __name__ == "__main__":
    main()

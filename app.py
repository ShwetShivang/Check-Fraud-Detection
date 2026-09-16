"""
Cheque Fraud Detection — Streamlit Application

Professional web interface for:
  - Uploading cheque images
  - Running fraud detection with model selection
  - Displaying results with highlighted regions
  - Grad-CAM visualization
  - Model comparison dashboard
"""
import os
import sys
import io
import numpy as np
import cv2
import torch
import streamlit as st
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Setup paths
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from src.fraud_pipeline import FraudDetector
from src.data_loader import load_dataset, SyntheticChequeGenerator
from src.preprocessing import preprocess_cheque


# ─── Page Config ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="Cheque Fraud Detection",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS ───────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Main theme */
    .main .block-container {
        padding-top: 2rem;
        max-width: 1200px;
    }
    
    /* Header gradient */
    .header-gradient {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 2rem 2.5rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
        border: 1px solid rgba(255,255,255,0.05);
    }
    .header-gradient h1 {
        color: #e94560;
        font-size: 2.2rem;
        margin-bottom: 0.3rem;
        font-weight: 700;
    }
    .header-gradient p {
        color: #a0aec0;
        font-size: 1.05rem;
        margin: 0;
    }
    
    /* Result cards */
    .result-card {
        background: linear-gradient(145deg, #1e293b, #0f172a);
        border-radius: 14px;
        padding: 1.8rem;
        margin: 1rem 0;
        border: 1px solid rgba(255,255,255,0.08);
        box-shadow: 0 4px 20px rgba(0,0,0,0.25);
    }
    
    /* Verdict styles */
    .verdict-fraud {
        color: #ef4444;
        font-size: 2.5rem;
        font-weight: 800;
        text-transform: uppercase;
        text-shadow: 0 0 20px rgba(239,68,68,0.5);
        letter-spacing: 0.05em;
    }
    .verdict-genuine {
        color: #22c55e;
        font-size: 2.5rem;
        font-weight: 800;
        text-transform: uppercase;
        text-shadow: 0 0 20px rgba(34,197,94,0.5);
        letter-spacing: 0.05em;
    }
    
    /* Confidence meter */
    .confidence-bar {
        height: 12px;
        border-radius: 6px;
        background: #1e293b;
        overflow: hidden;
        margin: 0.5rem 0;
    }
    .confidence-fill {
        height: 100%;
        border-radius: 6px;
        transition: width 0.5s ease;
    }
    
    /* Stats */
    .stat-box {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 10px;
        padding: 1rem 1.2rem;
        text-align: center;
    }
    .stat-number {
        font-size: 1.8rem;
        font-weight: 700;
        color: #e2e8f0;
    }
    .stat-label {
        font-size: 0.85rem;
        color: #94a3b8;
        margin-top: 0.3rem;
    }
    
    /* Sidebar */
    .css-1d391kg {
        background: #0f172a;
    }
    
    /* Upload area */
    .stFileUploader > div {
        border: 2px dashed #334155;
        border-radius: 12px;
        background: rgba(15, 23, 42, 0.5);
    }
    
    /* Button */
    .stButton > button {
        background: linear-gradient(135deg, #e94560, #c72c41);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        font-size: 1.1rem;
        font-weight: 600;
        border-radius: 10px;
        width: 100%;
        transition: all 0.3s;
        box-shadow: 0 4px 15px rgba(233,69,96,0.3);
    }
    .stButton > button:hover {
        box-shadow: 0 6px 25px rgba(233,69,96,0.5);
        transform: translateY(-1px);
    }
    
    /* Section headers */
    .section-header {
        color: #e2e8f0;
        font-size: 1.3rem;
        font-weight: 600;
        margin-top: 1.5rem;
        margin-bottom: 0.8rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid rgba(233,69,96,0.3);
    }
    
    hr {
        border-color: rgba(255,255,255,0.06);
    }
</style>
""", unsafe_allow_html=True)


# ─── Header ──────────────────────────────────────────────────────────
st.markdown("""
<div class="header-gradient">
    <h1>🔍 Cheque Fraud Detection</h1>
    <p>AI-powered ink variation analysis to detect fraudulent modifications in cheque images</p>
</div>
""", unsafe_allow_html=True)


# ─── Sidebar ─────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    st.markdown("---")
    
    model_type = st.selectbox(
        "Detection Model",
        ['siamese', 'cnn', 'classical'],
        format_func=lambda x: {
            'siamese': '🧠 Siamese Network (Recommended)',
            'cnn': '📊 CNN Baseline',
            'classical': '🔬 Classical ML (SVM/RF)'
        }[x],
        help="Choose the model for fraud detection"
    )
    
    st.markdown("---")
    st.markdown("### 📋 Detection Parameters")
    
    sim_threshold = st.slider(
        "Similarity Threshold",
        0.1, 0.9, config.SIMILARITY_THRESHOLD,
        0.05,
        help="Patches below this threshold are flagged as suspicious"
    )
    
    fraud_threshold = st.slider(
        "Fraud Region Threshold",
        0.05, 0.5, config.FRAUD_REGION_THRESHOLD,
        0.05,
        help="If more than this fraction of patches are suspicious → FRAUD"
    )
    
    st.markdown("---")
    st.markdown("### 📖 About")
    st.markdown("""
    This system detects cheque fraud by analyzing 
    ink variations across different regions.
    
    **Methodology:**
    - Patch-based comparison
    - Siamese similarity learning
    - LAB color space analysis
    
    **Reference:**  
    Dansena et al., ICPRMI 2017
    """)
    
    st.markdown("---")
    st.markdown("### 🏗️ Model Status")
    for model_name, filename in [
        ("Siamese", "siamese_best.pth"),
        ("CNN", "cnn_best.pth"),
        ("Classical SVM", "classical_svm.pkl"),
    ]:
        path = os.path.join(config.MODEL_DIR, filename)
        if os.path.exists(path):
            st.markdown(f"✅ {model_name}")
        else:
            st.markdown(f"⚠️ {model_name} (not trained)")


# ─── Main Content ────────────────────────────────────────────────────

# File uploader
col_upload, col_sample = st.columns([2, 1])

with col_upload:
    st.markdown('<p class="section-header">📤 Upload Cheque Image</p>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Drag and drop or browse",
        type=['jpg', 'jpeg', 'png', 'bmp', 'tif', 'tiff'],
        label_visibility="collapsed"
    )

with col_sample:
    st.markdown('<p class="section-header">🖼️ Sample Images</p>', unsafe_allow_html=True)
    # List available sample cheque images
    sample_dir = os.path.join(config.BASE_DIR, "data", "samples")
    sample_files = []
    if os.path.isdir(sample_dir):
        sample_files = sorted([
            f for f in os.listdir(sample_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'))
        ])
    if sample_files:
        sample_choice = st.selectbox(
            "Select a sample cheque",
            sample_files,
            format_func=lambda x: x.replace('_', ' ').rsplit('.', 1)[0].title(),
            label_visibility="collapsed"
        )
        use_sample = st.button("Load Sample Cheque")
    else:
        st.caption("No sample images found in data/samples/")
        use_sample = False
        sample_choice = None


# Initialize session state
if 'sample_image' not in st.session_state:
    st.session_state.sample_image = None

# Process input
image = None
image_source = None

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_source = "uploaded"
    st.session_state.sample_image = None  # Clear sample when file uploaded

elif use_sample and sample_choice:
    # Load the selected realistic sample cheque image
    sample_path = os.path.join(sample_dir, sample_choice)
    img = cv2.imread(sample_path)
    if img is not None:
        st.session_state.sample_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    image = st.session_state.sample_image
    image_source = "sample"

elif st.session_state.sample_image is not None:
    image = st.session_state.sample_image
    image_source = "sample"

# ─── Display & Detection ─────────────────────────────────────────────
if image is not None:
    st.markdown("---")
    
    col_img, col_ctrl = st.columns([3, 1])
    
    with col_img:
        st.markdown('<p class="section-header">📄 Input Cheque</p>', unsafe_allow_html=True)
        st.image(image, use_container_width=True)
    
    with col_ctrl:
        st.markdown('<p class="section-header">🎯 Actions</p>', unsafe_allow_html=True)
        st.markdown("")
        detect_btn = st.button("🔍 Detect Fraud", use_container_width=True)
        
        st.markdown("")
        show_gradcam = st.checkbox("Show Grad-CAM", value=True)
        show_sim_map = st.checkbox("Show Similarity Map", value=True)
        show_scores = st.checkbox("Show Score Distribution", value=True)
    
    # Run detection
    if detect_btn:
        # Update config thresholds
        config.SIMILARITY_THRESHOLD = sim_threshold
        config.FRAUD_REGION_THRESHOLD = fraud_threshold
        
        with st.spinner("🔄 Analyzing cheque image..."):
            try:
                detector = FraudDetector(model_type=model_type)
                result = detector.detect(image)
            except Exception as e:
                st.error(f"Detection error: {e}")
                result = None
        
        if result:
            st.markdown("---")
            
            # ── Verdict Banner ──
            verdict = result['verdict']
            confidence = result['confidence']
            
            if verdict == 'FRAUD':
                st.markdown(f"""
                <div class="result-card" style="border-left: 4px solid #ef4444;">
                    <p class="verdict-fraud">⚠️ FRAUD DETECTED</p>
                    <p style="color: #94a3b8; font-size: 1rem;">
                        Suspicious ink variations detected across cheque regions
                    </p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-card" style="border-left: 4px solid #22c55e;">
                    <p class="verdict-genuine">✅ GENUINE</p>
                    <p style="color: #94a3b8; font-size: 1rem;">
                        No significant ink variations detected
                    </p>
                </div>
                """, unsafe_allow_html=True)
            
            # ── Stats Row ──
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown(f"""
                <div class="stat-box">
                    <div class="stat-number">{confidence:.0%}</div>
                    <div class="stat-label">Confidence</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                num_scores = len(result['similarity_scores'])
                st.markdown(f"""
                <div class="stat-box">
                    <div class="stat-number">{num_scores}</div>
                    <div class="stat-label">Patch Pairs Analyzed</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                num_suspicious = len(result['suspicious_patches'])
                st.markdown(f"""
                <div class="stat-box">
                    <div class="stat-number">{num_suspicious}</div>
                    <div class="stat-label">Suspicious Regions</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col4:
                avg_sim = np.mean(result['similarity_scores']) if result['similarity_scores'] else 1.0
                st.markdown(f"""
                <div class="stat-box">
                    <div class="stat-number">{avg_sim:.2f}</div>
                    <div class="stat-label">Avg. Similarity</div>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("")
            
            # ── Highlighted Image ──
            col_orig, col_high = st.columns(2)
            
            with col_orig:
                st.markdown('<p class="section-header">📄 Original</p>', unsafe_allow_html=True)
                st.image(image, use_container_width=True)
            
            with col_high:
                st.markdown('<p class="section-header">🔴 Suspicious Regions</p>', unsafe_allow_html=True)
                if result['highlighted_image'] is not None:
                    st.image(result['highlighted_image'], use_container_width=True)
            
            # ── Similarity Map ──
            if show_sim_map and result['similarity_map'] is not None:
                st.markdown('<p class="section-header">🌡️ Similarity Heatmap</p>', unsafe_allow_html=True)
                
                fig, ax = plt.subplots(figsize=(10, 4))
                im = ax.imshow(result['similarity_map'], cmap='RdYlGn',
                              vmin=0, vmax=1, aspect='auto')
                ax.set_title('Ink Similarity Map (Green=Same, Red=Different)', fontsize=12)
                ax.axis('off')
                fig.colorbar(im, ax=ax, fraction=0.02, pad=0.04,
                           label='Similarity Score')
                fig.tight_layout()
                st.pyplot(fig)
                plt.close()
            
            # ── Score Distribution ──
            if show_scores and result['similarity_scores']:
                st.markdown('<p class="section-header">📊 Similarity Score Distribution</p>', 
                           unsafe_allow_html=True)
                
                fig, ax = plt.subplots(figsize=(8, 4))
                scores = result['similarity_scores']
                ax.hist(scores, bins=20, color='#3b82f6', alpha=0.7, edgecolor='#1e3a5f')
                ax.axvline(x=sim_threshold, color='#ef4444',
                          linestyle='--', linewidth=2, label=f'Threshold ({sim_threshold})')
                ax.set_xlabel('Similarity Score')
                ax.set_ylabel('Frequency')
                ax.set_title('Distribution of Pairwise Similarity Scores')
                ax.legend()
                ax.grid(True, alpha=0.3)
                fig.tight_layout()
                st.pyplot(fig)
                plt.close()
            
            # ── Grad-CAM ──
            if show_gradcam and result.get('gradcam_overlay') is not None:
                st.markdown('<p class="section-header">🧠 Grad-CAM Visualization</p>', 
                           unsafe_allow_html=True)
                st.image(result['gradcam_overlay'], caption="Model attention heatmap on a sample patch",
                        use_container_width=False, width=300)

            # ── Note ──
            if result.get('note'):
                st.info(result['note'])

else:
    # Landing state
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="stat-box">
            <div style="font-size: 2rem;">📤</div>
            <div class="stat-label" style="margin-top: 0.5rem; font-size: 1rem;">
                <strong>Step 1:</strong> Upload Image
            </div>
            <div class="stat-label">Upload a cheque image or use a sample</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="stat-box">
            <div style="font-size: 2rem;">🔍</div>
            <div class="stat-label" style="margin-top: 0.5rem; font-size: 1rem;">
                <strong>Step 2:</strong> Detect Fraud
            </div>
            <div class="stat-label">Click the detect button to analyze</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="stat-box">
            <div style="font-size: 2rem;">📊</div>
            <div class="stat-label" style="margin-top: 0.5rem; font-size: 1rem;">
                <strong>Step 3:</strong> View Results
            </div>
            <div class="stat-label">Review fraud analysis and highlighted regions</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("")
    st.info("👆 Upload a cheque image above or click **Generate Sample Cheque** to get started.")


# ─── Footer ──────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #64748b; font-size: 0.85rem;">
    <p>Cheque Fraud Detection using Ink Variation Analysis</p>
    <p>Based on: Dansena, Bag & Pal — "Differentiating Pen Inks in Hand-written Bank Cheques", ICPRMI 2017</p>
</div>
""", unsafe_allow_html=True)

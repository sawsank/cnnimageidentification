import streamlit as st
import os
from PIL import Image
import cv2
import numpy as np
import tempfile
from face_engine import FaceEngine

# Page Config
st.set_page_config(
    page_title="Biometric ID | FaceGuard",
    page_icon="👤",
    layout="wide"
)

# Advanced CSS for Premium Look (Glassmorphism + Modern UI)
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    
    html, body, [class*="css"]  {
        font-family: 'Inter', sans-serif;
    }
    
    .stApp {
        background: radial-gradient(circle at top right, #1e293b, #0f172a);
    }
    
    /* Glassmorphism containers */
    div.stVerticalBlock > div[style*="flex-direction: column"] > div {
        background: rgba(30, 41, 59, 0.7);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        padding: 20px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        margin-bottom: 20px;
    }
    
    /* Header optimization */
    .main-title {
        font-weight: 700;
        background: linear-gradient(to right, #4ade80, #3b82f6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem;
        margin-bottom: 0px;
    }
    
    /* Tab Styling */
    .stTabs [data-baseweb="tab-list"] {
        background-color: transparent;
        gap: 10px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: rgba(255, 255, 255, 0.05);
        border-radius: 8px 8px 0 0;
        padding: 10px 20px;
        color: #94a3b8;
        border: none;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: rgba(74, 222, 128, 0.2);
        color: #4ade80 !important;
        border-bottom: 2px solid #4ade80 !important;
    }
    
    /* Banner Styles */
    .res-banner {
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        font-weight: 600;
        letter-spacing: 0.5px;
        margin-bottom: 1rem;
    }
    .match { background-color: rgba(34, 197, 94, 0.2); border: 1px solid #22c55e; color: #4ade80; }
    .no-match { background-color: rgba(239, 68, 68, 0.2); border: 1px solid #ef4444; color: #f87171; }
    
    /* Metric styling */
    [data-testid="stMetricValue"] {
        color: #4ade80;
    }
    </style>
    """, unsafe_allow_html=True)

# Header
st.markdown('<h1 class="main-title">FaceGuard Pro</h1>', unsafe_allow_html=True)
st.markdown("<p style='color: #94a3b8; font-size: 1.2rem;'>Secure Biometric Identity & Recognition Suite</p>", unsafe_allow_html=True)

# Initialize Engine
@st.cache_resource
def get_engine():
    return FaceEngine()

engine = get_engine()

tab1, tab2 = st.tabs(["🔍 Identification Engine", "📥 Enrollment Portal"])

# --- TAB 1: RECOGNITION ---
with tab1:
    col1, col2 = st.columns([1, 1], gap="large")
    
    with col1:
        st.markdown("### 📸 Biometric Input")
        choice = st.segmented_control("Capture Method", ["Web Cam", "Upload Image"], default="Web Cam")
        
        input_img = None
        if choice == "Web Cam":
            input_img = st.camera_input("Scanner Interface")
        else:
            input_img = st.file_uploader("Drop enrollment image here", type=["jpg", "png", "jpeg"])
            
    with col2:
        st.markdown("### 🔬 Neural Analysis")
        if input_img:
            # Save temporary image for processing
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
                tmp.write(input_img.getvalue())
                tmp_path = tmp.name
            
            with st.status("Running Biometric Analysis...", expanded=True) as status:
                st.write("Initializing Neural Network...")
                st.write("Extracting facial landmarks...")
                result = engine.recognize_face(tmp_path)
                st.write("Comparing embeddings against secure database...")
                status.update(label="Analysis Complete!", state="complete", expanded=False)
            
            if result["found"]:
                st.markdown(f'<div class="res-banner match">VERIFIED IDENTITY: {result["name"].upper()}</div>', unsafe_allow_html=True)
                
                m1, m2 = st.columns(2)
                m1.metric("Similarity Score", f"{result['distance']:.4f}")
                m2.metric("Confidence", f"{max(0, 100 - result['distance']*100):.1f}%")
                
                # Display the database image
                db_img_path = os.path.join("database", f"{result['name']}.jpg")
                if os.path.exists(db_img_path):
                    st.image(db_img_path, caption=f"Database Reference: {result['name']}", width=250)
            else:
                st.markdown('<div class="res-banner no-match">IDENTITY NOT VERIFIED</div>', unsafe_allow_html=True)
                st.error("No matching biometric patterns found in the secure registry.")
            
            os.remove(tmp_path)
        else:
            st.info("Scanner Ready. Please provide an input source.")

# --- TAB 2: ENROLLMENT ---
with tab2:
    e_col1, e_col2 = st.columns([1, 1], gap="large")
    
    with e_col1:
        st.markdown("### 📥 Profile Enrollment")
        st.write("Register a new individual into the secure biometric database.")
        
        enroll_name = st.text_input("Full Name Identifier", placeholder="Enter name...")
        enroll_img = st.file_uploader("Clear Portrait Portrait", type=["jpg", "png", "jpeg"], key="enroll_upload")
        
        if st.button("Begin Enrollment Sequence", width='stretch'):
            if enroll_name and enroll_img:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
                    tmp.write(enroll_img.getvalue())
                    tmp_path = tmp.name
                
                with st.spinner("Processing Bio-Enrolment..."):
                    saved_path = engine.enroll_face(tmp_path, enroll_name)
                
                os.remove(tmp_path)
                st.success(f"Identity **{enroll_name}** successfully registered.")
                st.balloons()
            else:
                st.warning("Identification data incomplete. Name and Portrait required.")

    with e_col2:
        st.markdown("### 📂 Secure Registry")
        db_images = [f for f in os.listdir("database") if f.endswith((".jpg", ".png"))]
        if db_images:
            cols = st.columns(3)
            for i, img_name in enumerate(db_images):
                name = os.path.splitext(img_name)[0]
                cols[i % 3].image(os.path.join("database", img_name), caption=name, width="stretch")
        else:
            st.info("Registry is currently empty.")

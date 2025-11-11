
import streamlit as st
import pandas as pd
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
import time
import whisper
from ultralytics import YOLO
from PIL import Image
import json
import re
import numpy as np

APP_TITLE = "By-Law Operations Co-Pilot (PoC)"
MOCK_DATA_FILE = "complaints.csv"
MOCK_AUDIO_FILE = "data/sample_call.m4a"
MOCK_IMAGE_FILE = "data/sample_violation.jpg"
MODEL_FILE = "data/pothole_model.pt"

# --- 1. Global Configs & API Key ---
st.set_page_config(
    page_title=f"Proactive AI | {APP_TITLE}",
    page_icon="✨",
    layout="wide"
)

# Load secrets
GOOGLE_API_KEY = st.secrets.get("GEMINI_API_KEY")
if not GOOGLE_API_KEY:
    st.error("GEMINI_API_KEY secret not found. Please set it in your Streamlit app settings.")
    st.stop()
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

GEMINI_LLM_MODEL = "models/gemini-pro-latest"

# --- 2. GenAI Prompt Template (from Day 1) ---
INTAKE_PROMPT_TEMPLATE = """
You are an AI data entry assistant.
Your task is to analyze the following 311 call transcript and extract key entities.
Respond *only* with a valid JSON object.

---
TRANSCRIPT:
{transcript}
---

JSON_OUTPUT:
"""

# --- 3. Caching & Model Loading Functions ---
@st.cache_resource
def load_llm():
    try:
        llm = ChatGoogleGenerativeAI(model=GEMINI_LLM_MODEL)
        return llm
    except Exception as e:
        st.error(f"Error initializing Gemini: {e}")
        st.stop()

@st.cache_resource
def load_whisper_model():
    # Load the base model. It's small, fast, and good enough for a demo.
    try:
        model = whisper.load_model("base")
        return model
    except Exception as e:
        st.error(f"Error loading Whisper model: {e}")
        st.stop()

@st.cache_resource
def load_yolo_model():
    """Loads the custom-trained YOLOv8 model."""
    if not os.path.exists(MODEL_FILE):
        st.error(f"Model file not found: {MODEL_FILE}. Please ensure it's in the repo 'data' folder.")
        return None
    try:
        model = YOLO(MODEL_FILE)
        return model
    except Exception as e:
        st.error(f"Error loading YOLO model: {e}")
        return None

# Load all models
llm = load_llm()
whisper_model = load_whisper_model()
yolo_model = load_yolo_model()

# --- 4. Streamlit App UI ---
st.title("✨ Advanced Features: Proactive AI Detection")
st.markdown("This page demonstrates the advanced 'proactive' capabilities from our research, showing how AI can *generate* new, high-value data, not just analyze existing complaints.")

tab1, tab2, tab3 = st.tabs([
    "🎙️ **AI Call Intake (Auto311)**",
    "📸 **AI Violation Detection (Stockton Model)**",
    "🗺️ **AI Hotspotting (Simulation)**"
])

# --- TAB 1: AI Call Intake ---
with tab1:
    st.header("AI 311 Call Intake")
    st.markdown("This simulates the 'Auto311' system. Upload a sample audio complaint (like the `.m4a` from the repo's `data` folder) to see the AI transcribe and auto-fill the case file.")
    
    uploaded_audio = st.file_uploader("Upload a 311 audio complaint (.m4a, .mp3, .wav)...", type=["m4a", "mp3", "wav"])
    
    if uploaded_audio is not None:
        st.audio(uploaded_audio, format='audio/m4a')
        
        if st.button("Transcribe & Extract Entities", use_container

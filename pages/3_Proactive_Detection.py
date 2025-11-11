
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
st.title("✨ 'Wow' Features: Proactive AI Detection")
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
        
        if st.button("Transcribe & Extract Entities", use_container_width=True):
            with st.spinner("Transcribing audio with Whisper..."):
                # Save temp file
                with open("temp_audio.m4a", "wb") as f:
                    f.write(uploaded_audio.getbuffer())
                
                result = whisper_model.transcribe("temp_audio.m4a")
                transcript = result['text']
            
            st.subheader("1. AI-Generated Transcript")
            st.info(transcript)
            
            with st.spinner("Extracting entities with Gemini..."):
                try:
                    prompt_int = ChatPromptTemplate.from_template(INTAKE_PROMPT_TEMPLATE)
                    chain_int = prompt_int | llm
                    response_int = chain_int.invoke({"transcript": transcript})
                    
                    # Clean the JSON output from markdown
                    json_match = re.search(r"```json\n([\s\S]*?)\n```", response_int.content, re.IGNORECASE)
                    if json_match:
                        json_str = json_match.group(1)
                    else:
                        json_str = response_int.content

                    st.subheader("2. Auto-Filled Complaint Form (JSON)")
                    st.json(json_str)
                except Exception as e:
                    st.error(f"API Error. Free tier quota may be exceeded. Please wait a minute and try again. {e}")
            
            os.remove("temp_audio.m4a")

# --- TAB 2: AI Violation Detection ---
with tab2:
    st.header("Proactive CV Detection")
    st.markdown("This simulates the 'Stockton Model'. A city vehicle with a camera uses AI to find violations *without* a 311 call. Upload an image (like the `sample_violation.jpg` from the repo's `data` folder).")

    uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        
        if yolo_model:
            with st.spinner("Analyzing image for violations..."):
                results = yolo_model(image)
                plotted_image = results[0].plot() # .plot() draws boxes on the image
                
            st.subheader("Processed Image with Detections")
            st.image(plotted_image, caption="AI Detection (Model: YOLOv8-Pothole)", use_column_width=True)
            
            st.subheader("AI-Generated Proactive Case")
            st.success("🤖 **Proactive Case Opened:** 'Illegal Debris/Pothole' detected at GPS 43.68, -79.75. Assigning to 'Property Standards' queue.")
        else:
            st.error("YOLO model not loaded. Check app logs.")

# --- TAB 3: AI Hotspotting Simulation ---
with tab3:
    st.header("AI Predictive Hotspotting (Simulation)")
    st.markdown("This simulates the 'data flywheel' concept. A predictive model fed with *only* 311 data is biased. Adding proactive CV data makes it smarter.")
    
    # Load data for mapping
    df = st.session_state.get('df')
    if df is not None:
        st.subheader("Data Source Selection")
        data_source = st.selectbox(
            "Select data source for predictive map:",
            ("1. 311 Complaint Data Only (Biased)", "2. 311 + Proactive CV Data (Unbiased)")
        )
        
        if data_source == "1. 311 Complaint Data Only (Biased)":
            st.warning("**Showing Biased Data:** These hotspots are based only on where residents *report* issues (the 'squeaky wheel' problem). This model over-prioritizes some areas and misses others entirely.")
            map_df = df[df['Status'] != 'Closed'][['GPS_Lat', 'GPS_Lon']]
            map_df.rename(columns={'GPS_Lat': 'lat', 'GPS_Lon': 'lon'}, inplace=True)
            st.map(map_df, zoom=10)
        else:
            st.success("**Showing Unbiased Data:** This map is smarter. It combines 311 data with the *proactive CV data* from our vehicles. This model finds violations city-wide, not just where people complain, leading to fairer and more efficient enforcement.")
            # We'll just add some random noise to the coordinates to "simulate" the new data
            map_df = df[df['Status'] != 'Closed'][['GPS_Lat', 'GPS_Lon']]
            map_df['GPS_Lat'] = map_df['GPS_Lat'] + (np.random.randn(len(map_df)) * 0.01)
            map_df.rename(columns={'GPS_Lat': 'lat', 'GPS_Lon': 'lon'}, inplace=True)
            st.map(map_df, zoom=10)
    else:
        st.info("Load the Analytics Dashboard page first to initialize the data.")

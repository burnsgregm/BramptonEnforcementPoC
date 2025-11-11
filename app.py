
import streamlit as st
import pandas as pd
import os

APP_TITLE = "By-Law Operations Co-Pilot (PoC)"
MOCK_DATA_FILE = "complaints.csv"

st.set_page_config(
    page_title=f"{APP_TITLE} - Home",
    page_icon="🤖",
    layout="wide"
)

st.title(f"Welcome to the {APP_TITLE}")
st.markdown("""
This multi-page app is a functional prototype for the City of Brampton's AI PoC Program.
It demonstrates a "best-in-class" solution for the **Proactive Enforcement Response** challenge.

This PoC simulates the entire "data flywheel":
1.  **Analytics (Part A):** A live dashboard for analyzing and prioritizing complaint trends.
2.  **Generation (Part B):** GenAI tools to automate officer workflows and generate resident-friendly messages.
3.  **"Wow" Features:** Proactive AI tools for CV and audio intake.

---
""")

st.header("How to use this demo:")
st.page_link("pages/1_Analytics_Dashboard.py", label="**1. Analytics Dashboard**", icon="📊")
st.markdown("View a simulated, live dashboard of all by-law complaints. Analyze complaint types, see hotspots on a map, and use the interactive table to filter data.")

st.page_link("pages/2_Case_Management.py", label="**2. Case Management Workflow**", icon="👩‍⚖️")
st.markdown("Select a complaint from the list to see the GenAI workflow. Generate an AI-powered 'on-site briefing' and draft a 'resident-friendly' closure message.")

st.page_link("pages/3_Proactive_Detection.py", label="**3. Proactive Detection**", icon="✨")
st.markdown("See the 'proactive' AI features in action, including transcribing 311 calls with Whisper, extracting data with Gemini, and detecting violations with a YOLOv8 Computer Vision model.")

st.header("Mock Data")
st.markdown(f"This entire demo runs on a 100-row mock database (`{MOCK_DATA_FILE}`) that is loaded into the app's memory.")

if os.path.exists(MOCK_DATA_FILE):
    df = pd.read_csv(MOCK_DATA_FILE)
    st.dataframe(df.head(), use_container_width=True)
else:
    st.error(f"{MOCK_DATA_FILE} not found. Please ensure it's in the repo.")

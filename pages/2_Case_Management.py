
import streamlit as st
import pandas as pd
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
import time

APP_TITLE = "By-Law Operations Co-Pilot (PoC)"
MOCK_DATA_FILE = "complaints.csv"

# --- 1. Global Configs & API Key ---
st.set_page_config(
    page_title=f"Case Management | {APP_TITLE}",
    page_icon="👩‍⚖️",
    layout="wide"
)

# Load secrets
GOOGLE_API_KEY = st.secrets.get("GEMINI_API_KEY")
if not GOOGLE_API_KEY:
    st.error("GEMINI_API_KEY secret not found. Please set it in your Streamlit app settings.")
    st.stop()
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

# Get model name from Colab
GEMINI_LLM_MODEL = "models/gemini-pro-latest"

# --- 2. GenAI Prompt Templates (from Day 1) ---
GENERATOR_PROMPT_TEMPLATE = """
You are a communication assistant for the City of Brampton By-Law Services.
Your tone is professional, empathetic, and clear (6th-grade reading level).
Your task is to write a resident-friendly closure message based on an officer's internal report.
RULES:
1.  **BE POLITE:** Always thank the resident for their report.
2.  **BE CLEAR:** State the official finding clearly, in plain language.
3.  **DO NOT SHARE INTERNAL NOTES:** Never, under any circumstances, include the "Officer's Private Notes." Only use the "Official Finding."
---
COMPLAINT ID: {complaint_id}
OFFICIAL FINDING: {official_finding}
OFFICER'S PRIVATE NOTES: {private_notes}
---
Generate the resident-friendly message below:
"""

SUMMARIZER_PROMPT_TEMPLATE = """
You are an AI assistant for a By-Law Officer.
Your task is to summarize a long, complex case history into a 1-paragraph "on-site briefing".
Focus only on the most critical facts: repeat offenses, officer notes, and any mention of threats or aggression.
---
CASE HISTORY:
{case_history}
---
Generate the 1-paragraph "On-Site Briefing" below:
"""

# --- 3. Caching & GenAI Functions ---
@st.cache_resource
def load_llm():
    try:
        llm = ChatGoogleGenerativeAI(model=GEMINI_LLM_MODEL)
        return llm
    except Exception as e:
        st.error(f"Error initializing Gemini: {e}")
        st.stop()

@st.cache_data
def load_data(filepath):
    if not os.path.exists(filepath):
        return None
    df = pd.read_csv(filepath)
    return df

llm = load_llm()

# --- 4. Streamlit App UI ---
st.title("👩‍⚖️ Case Management Workflow")
st.markdown("This page demonstrates **Part B** of the PoC: using Generative AI to automate officer workflows and 'generate personalized, resident-friendly closure messages'.")

# Load data
df = load_data(MOCK_DATA_FILE)
if df is None:
    st.error(f"{MOCK_DATA_FILE} not found. Please run the Colab notebook to generate it.")
    st.stop()

# --- 1. Complaint Selector ---
st.header("Step 1: Select a Complaint")
complaint_ids = df['ComplaintID'].tolist()
selected_id = st.selectbox("Select a Complaint ID to manage:", complaint_ids)

if not selected_id:
    st.stop()

# Get the data for the selected complaint
case_data = df[df['ComplaintID'] == selected_id].iloc[0]

# Display the selected case data in a clean way
st.subheader(f"Managing Complaint: {case_data['ComplaintID']}")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Status", case_data['Status'])
col2.metric("Type", case_data['Type'])
col3.metric("Ward", case_data['Ward'])
col4.metric("Date", case_data['Date'])

st.divider()

# --- 2. AI Summarizer (Officer Briefing) ---
st.header("Step 2: Get AI On-Site Briefing")
st.markdown("This AI tool reads the *entire* complex case history and generates a 1-paragraph summary for the officer.")

with st.expander("Show Full Case History (Internal Only)"):
    st.text(case_data['CaseHistory (Internal)'])

if st.button("Generate Officer Briefing", use_container_width=True):
    with st.spinner("AI is summarizing the case..."):
        try:
            prompt_sum = ChatPromptTemplate.from_template(SUMMARIZER_PROMPT_TEMPLATE)
            chain_sum = prompt_sum | llm
            response_sum = chain_sum.invoke({"case_history": case_data['CaseHistory (Internal)']})
            
            st.info(response_sum.content)
        except Exception as e:
            st.error(f"API Error. Free tier quota may be exceeded. Please wait a minute and try again. {e}")

st.divider()

# --- 3. AI Generator (Closure Message) ---
st.header("Step 3: Action Complaint & Generate Closure Message")
st.markdown("Select an official finding and let the AI generate the 'personalized, resident-friendly' message.")

col1, col2 = st.columns(2)
with col1:
    st.subheader("Officer Action (Internal)")
    official_finding = st.selectbox("Select Official Finding:", 
                                    ["Violation found, warning issued.", 
                                     "Violation found, ticket issued.", 
                                     "No violation found at time of visit.",
                                     "Duplicate complaint, already resolved."])
    
    st.text_area("Officer's Private Notes (Internal Only):", 
                 value=case_data['OfficerNotes (Internal)'], 
                 height=150)

with col2:
    st.subheader("AI-Generated Message (External)")
    if st.button("Generate Resident-Friendly Message", use_container_width=True):
        with st.spinner("AI is drafting the message..."):
            try:
                prompt_gen = ChatPromptTemplate.from_template(GENERATOR_PROMPT_TEMPLATE)
                chain_gen = prompt_gen | llm
                
                response_gen = chain_gen.invoke({
                    "complaint_id": case_data['ComplaintID'],
                    "official_finding": official_finding,
                    "private_notes": case_data['OfficerNotes (Internal)'] # The AI is instructed not to use this
                })
                
                st.text_area("Generated Message:", value=response_gen.content, height=250)
            except Exception as e:
                st.error(f"API Error. Free tier quota may be exceeded. Please wait a minute and try again. {e}")

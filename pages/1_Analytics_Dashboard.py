
import streamlit as st
import pandas as pd
import os

APP_TITLE = "By-Law Operations Co-Pilot (PoC)"
MOCK_DATA_FILE = "complaints.csv"

st.set_page_config(
    page_title=f"Analytics | {APP_TITLE}",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Analytics & Prioritization Dashboard")
st.markdown("This dashboard simulates **Part A** of the PoC: 'analyzing... complaint trends' to 'support informed decisions and prioritization'.")

# --- Data Loading ---
@st.cache_data
def load_data(filepath):
    if not os.path.exists(filepath):
        st.error(f"Data file not found: {filepath}")
        return None
    df = pd.read_csv(filepath)
    df['Date'] = pd.to_datetime(df['Date'])
    return df

df = load_data(MOCK_DATA_FILE)

if df is None:
    st.stop()

# Store in session state for cross-page use
st.session_state.df = df

# --- 1. Live Metrics ---
st.header("Live Operational Metrics")
col1, col2, col3 = st.columns(3)
col1.metric("Total Open Complaints", df[df['Status'] != 'Closed'].shape[0])
col2.metric("Top Complaint Type", df['Type'].mode()[0])
col3.metric("Busiest Ward", f"Ward {df['Ward'].mode()[0]}")

st.divider()

# --- 2. Analytics & Hotspot Map ---
st.header("Complaint Trend Analysis")
col1, col2 = st.columns([0.4, 0.6]) # 40% width for charts, 60% for map

with col1:
    st.subheader("Complaints by Type")
    st.bar_chart(df['Type'].value_counts())
    
    st.subheader("Complaints by Ward")
    st.bar_chart(df['Ward'].value_counts())

with col2:
    st.subheader("Complaint Hotspot Map")
    st.markdown("This simulates the 'Predictive Hotspot' feature.")
    # We only map 'Pending' or 'In Progress' complaints
    map_df = df[df['Status'] != 'Closed'][['GPS_Lat', 'GPS_Lon']]
    map_df.rename(columns={'GPS_Lat': 'lat', 'GPS_Lon': 'lon'}, inplace=True)
    st.map(map_df, zoom=10)

st.divider()

# --- 3. Interactive Data Table ---
st.header("Full Complaint Log")
st.markdown("Use this interactive table to search, filter, and sort all complaints.")

# Using st.data_editor gives us the interactive, editable table.
# For this demo, we'll disable editing on most columns.
edited_df = st.data_editor(
    df,
    use_container_width=True,
    num_rows="dynamic",
    column_config={
        "ComplaintID": st.column_config.TextColumn(disabled=True),
        "Type": st.column_config.TextColumn(disabled=True),
        "Ward": st.column_config.TextColumn(disabled=True),
        "OfficerNotes (Internal)": st.column_config.TextColumn(disabled=True),
        "CaseHistory (Internal)": st.column_config.TextColumn(disabled=True),
    },
    hide_index=True
)

# This is the "action" part. If a user changes a status (e.g., to "Closed")
# and we had a real database, we would write the change here.
# For the demo, we just show it's possible.
st.success("You can edit the 'Status' or 'Date' fields directly in the table.")

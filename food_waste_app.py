import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import os

# --- 1. PAGE SETUP ---
st.set_page_config(page_title="Eco-Feed AI", layout="wide", page_icon="🥗")

# --- 2. DATA LOADING (From File) ---
def load_data():
    file_path = 'food_data.csv'
    if os.path.exists(file_path):
        return pd.read_csv(file_path)
    else:
        return None

df = load_data()

# --- 3. ERROR HANDLING (If file is missing) ---
if df is None:
    st.error("### ❌ Dataset File Missing!")
    st.info("""
    **To fix this:** Please ensure a file named `food_data.csv` is in the same folder as this code.
    
    **Your CSV should have these columns:**
    `day, students, menu_type, food_prepared, food_wasted`
    """)
    st.stop()

# --- 4. AI MODEL TRAINING ---
# We use Label Encoding to turn Text (Monday, Special) into Numbers (1, 0)
le_day = LabelEncoder().fit(df['day'])
le_menu = LabelEncoder().fit(df['menu_type'])

X = df.copy()
X['day_enc'] = le_day.transform(df['day'])
X['menu_enc'] = le_menu.transform(df['menu_type'])

# The AI learns how Day, Students, and Menu affect Waste
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X[['day_enc', 'students', 'menu_enc']], df['food_wasted'])

# --- 5. SIDEBAR INPUTS ---
st.sidebar.header("📋 Input Current Details")
input_day = st.sidebar.selectbox("Select Day", df['day'].unique())
input_students = st.sidebar.number_input("Expected Student Count", 100, 1000, 300)
input_menu = st.sidebar.selectbox("Select Menu Type", df['menu_type'].unique())

# --- 6. CORE CALCULATIONS ---
# AI Prediction
d_enc = le_day.transform([input_day])[0]
m_enc = le_menu.transform([input_menu])[0]
predicted_waste = model.predict([[d_enc, input_students, m_enc]])[0]

# Logic for Recommended Preparation
avg_food_per_person = (df['food_prepared'] - df['food_wasted']).mean() / df['students'].mean()
suggested_prep = (input_students * avg_food_per_person) + 5 # 5kg extra for safety

# --- 7. SIMPLE & CLEAN UI ---
st.title("🥗 Food Waste Management System")
st.markdown("---")

# Row 1: The Results
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Predicted Waste", f"{predicted_waste:.1f} KG")
    st.caption("Based on historical patterns")

with col2:
    st.metric("Suggested Preparation", f"{suggested_prep:.1f} KG")
    st.caption("Includes a 5kg safety buffer")

with col3:
    money_saved = predicted_waste * 60 # Assuming ₹60 per KG
    st.metric("Potential Savings", f"₹{money_saved:.0f}")
    st.caption("Financial impact today")

st.markdown("---")

# Row 2: Visual Evidence
v_col1, v_col2 = st.columns([1, 1])

with v_col1:
    st.subheader("📊 Historical Waste Trends")
    fig = px.bar(df, x='day', y='food_wasted', color='menu_type', barmode='group')
    st.plotly_chart(fig, use_container_width=True)

with v_col2:
    st.subheader("💡 Smart Recommendations")
    if input_day in ['Saturday', 'Sunday']:
        st.warning("📅 **Weekend Alert:** Attendance is usually lower. Reduce side-dishes.")
    if input_menu == "Special":
        st.info("🍽️ **Special Menu:** Higher interest expected. Prepare for fresh batches.")
    st.success("✅ **Goal:** Aim for less than 5% waste today.")

# Row 3: Audit Table
with st.expander("📂 View Source Data (food_data.csv)"):
    st.dataframe(df, use_container_width=True)
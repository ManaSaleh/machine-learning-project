# Import Libraries
import streamlit as st
import streamlit.components.v1 as components
import joblib
import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import OrdinalEncoder

# Set Page Configuration
st.set_page_config(page_title="Laptop Price Prediction & Workflow", page_icon="💻", layout="wide")

# Streamlit App Title
st.title("💻 Laptelligence Price Prediction")

# Tabs for Navigation
tabs = st.tabs(["🏠 Home", "🔧 Feature Selection", "📊 Predict Price", "ℹ️ About"])

# Load Model and Data
@st.cache_resource
def load_model():
    return joblib.load("Models/cat_model-v4.pkl")

@st.cache_data
def load_test_data():
    return pd.read_csv("Data/laptop_price - dataset.csv").drop('Price (Euro)', axis=1)

# Initialize
model = load_model()
df = load_test_data()
df = df.drop('Product', axis=1)
# ---- HOME TAB ----
with tabs[0]:
    st.header("Welcome to the Laptop Price Prediction App")
    st.markdown(
        """
        This app leverages **Machine Learning** to predict the price of a laptop based on its specifications.
        Explore the interactive workflow above to understand the process!
        """
    )

    # Enhanced HTML and SVG with animations and interactivity
    html_code = """
<div style="display: flex; justify-content: center; align-items: center; height: 70vh; background: radial-gradient(circle, #ffffff, #f8f9fa); border-radius: 15px;">
  <svg width="80%" height="80%" viewBox="0 0 700 700">
    <!-- Animated Paths -->
    <path d="M350,100 A300,300 0 0,1 600,350" stroke="rgba(150, 150, 150, 0.5)" stroke-width="3" fill="none">
      <animate attributeName="stroke-dasharray" from="0,600" to="600,0" dur="2s" repeatCount="indefinite" />
    </path>
    <path d="M600,350 A300,300 0 0,1 350,600" stroke="rgba(150, 150, 150, 0.5)" stroke-width="3" fill="none">
      <animate attributeName="stroke-dasharray" from="0,600" to="600,0" dur="2s" repeatCount="indefinite" />
    </path>
    <path d="M350,600 A300,300 0 0,1 100,350" stroke="rgba(150, 150, 150, 0.5)" stroke-width="3" fill="none">
      <animate attributeName="stroke-dasharray" from="0,600" to="600,0" dur="2s" repeatCount="indefinite" />
    </path>
    <path d="M100,350 A300,300 0 0,1 350,100" stroke="rgba(150, 150, 150, 0.5)" stroke-width="3" fill="none">
      <animate attributeName="stroke-dasharray" from="0,600" to="600,0" dur="2s" repeatCount="indefinite" />
    </path>

    <!-- Interactive Nodes -->
    <circle cx="350" cy="100" r="60" fill="url(#gradient1)" filter="url(#glow1)" />
    <text x="350" y="100" fill="#fff" font-size="30" text-anchor="middle" alignment-baseline="middle" font-family="Arial">📊</text>
    <text x="350" y="175" fill="#333" font-size="18" text-anchor="middle" font-family="Arial" font-weight="bold">Data</text>

    <circle cx="600" cy="350" r="60" fill="url(#gradient2)" filter="url(#glow2)" />
    <text x="600" y="350" fill="#fff" font-size="30" text-anchor="middle" alignment-baseline="middle" font-family="Arial">🤖</text>
    <text x="600" y="425" fill="#333" font-size="18" text-anchor="middle" font-family="Arial" font-weight="bold">Model</text>

    <circle cx="350" cy="600" r="60" fill="url(#gradient3)" filter="url(#glow3)" />
    <text x="350" y="600" fill="#fff" font-size="30" text-anchor="middle" alignment-baseline="middle" font-family="Arial">💡</text>
    <text x="350" y="675" fill="#333" font-size="18" text-anchor="middle" font-family="Arial" font-weight="bold">Insight</text>

    <circle cx="100" cy="350" r="60" fill="url(#gradient4)" filter="url(#glow4)" />
    <text x="100" y="350" fill="#fff" font-size="30" text-anchor="middle" alignment-baseline="middle" font-family="Arial">🚀</text>
    <text x="100" y="425" fill="#333" font-size="18" text-anchor="middle" font-family="Arial" font-weight="bold">Deploy</text>

    <!-- Gradient Backgrounds -->
    <defs>
      <radialGradient id="gradient1" cx="50%" cy="50%" r="50%">
        <stop offset="0%" style="stop-color: #f39c12; stop-opacity: 1;" />
        <stop offset="100%" style="stop-color: #f39c12; stop-opacity: 0.6;" />
      </radialGradient>
      <radialGradient id="gradient2" cx="50%" cy="50%" r="50%">
        <stop offset="0%" style="stop-color: #27ae60; stop-opacity: 1;" />
        <stop offset="100%" style="stop-color: #27ae60; stop-opacity: 0.6;" />
      </radialGradient>
      <radialGradient id="gradient3" cx="50%" cy="50%" r="50%">
        <stop offset="0%" style="stop-color: #3498db; stop-opacity: 1;" />
        <stop offset="100%" style="stop-color: #3498db; stop-opacity: 0.6;" />
      </radialGradient>
      <radialGradient id="gradient4" cx="50%" cy="50%" r="50%">
        <stop offset="0%" style="stop-color: #e74c3c; stop-opacity: 1;" />
        <stop offset="100%" style="stop-color: #e74c3c; stop-opacity: 0.6;" />
      </radialGradient>
    </defs>
  </svg>
</div>
"""
# Render the HTML in Streamlit
    components.html(html_code, height=500)



# ---- Preprocessing Function ----
def preprocess_input(data):
    # Memory Processing
    def memory_split(memory):
        try:
            if '+' in memory:
                mem1, mem2 = memory.split('+')
                mem_type = mem1.split(' ')[1] + '+' + mem2.split(' ')[1]
                mem1_capacity = int(re.findall(r'\d+', mem1)[0])
                mem2_capacity = int(re.findall(r'\d+', mem2)[0])
                if 'GB' in mem1 and 'GB' in mem2:
                    total_capacity = mem1_capacity + mem2_capacity
                elif 'TB' in mem1 and 'GB' in mem2:
                    total_capacity = mem1_capacity * 1024 + mem2_capacity
                elif 'GB' in mem1 and 'TB' in mem2:
                    total_capacity = mem1_capacity + mem2_capacity * 1024
                else:
                    total_capacity = mem1_capacity * 1024 + mem2_capacity * 1024
                return total_capacity, mem_type
            else:
                capacity = int(re.findall(r'\d+', memory)[0])
                mem_type = memory.split(' ')[1]
                if 'TB' in memory:
                    capacity *= 1024
                return capacity, mem_type
        except:
            return np.nan, np.nan

    # Screen Resolution Processing
    def process_screen_resolution(df):
        df[['Resolution_Width', 'Resolution_Height']] = df['ScreenResolution'].str.extract(r'(\d{3,4})x(\d{3,4})').astype(int)
        df['Contains_HD'] = df['ScreenResolution'].str.contains('HD', case=False).astype(int)
        df['Contains_IPS'] = df['ScreenResolution'].str.contains('IPS', case=False).astype(int)
        df['Contains_Touchscreen'] = df['ScreenResolution'].str.contains('Touchscreen', case=False).astype(int)
        df.drop(['ScreenResolution'], axis=1, inplace=True)

    # Extract CPU Features
    def extract_family(cpu_type, company):
        patterns = {
            "Intel": r'^(Core|Xeon|Pentium|Celeron|Atom|Core M)',
            "AMD": r'^(Ryzen|A[0-9]|FX|Athlon|E[0-9]|Pro|Sempron)',
            "Samsung": r'^(Exynos)',
        }
        match = re.search(patterns.get(company, 'Unknown'), cpu_type, re.IGNORECASE)
        return match.group(1) if match else 'Unknown'

    def extract_generation(cpu_type, company):
        if company == 'Intel':
            match = re.search(r'(\d{4,5}[A-Za-z]*)$', cpu_type)
        elif company == 'AMD':
            match = re.search(r'(\d{4,5})$', cpu_type)
        elif company == 'Samsung':
            match = re.search(r'Exynos (\d+)', cpu_type, re.IGNORECASE)
        else:
            match = None
        return match.group(1)[:1] if match else 'Unknown'

    # GPU Processing
    def process_gpu(df):
        df['GPU_Family'] = df['GPU_Type'].apply(lambda x: x.split(' ')[0])
        df['GPU_Series'] = df['GPU_Type'].str.extract(r'(\d+)').fillna(df['GPU_Type'])
        df.drop(['GPU_Type'], axis=1, inplace=True)

    # Apply Preprocessing Steps
    data['Memory Capacity'], data['Memory Type'] = zip(*data['Memory'].apply(memory_split))
    data.drop(['Memory'], axis=1, inplace=True)

    process_screen_resolution(data)

    data['CPU_Family'] = data.apply(lambda row: extract_family(row['CPU_Type'], row['CPU_Company']), axis=1)
    data['CPU_Generation'] = data.apply(lambda row: extract_generation(row['CPU_Type'], row['CPU_Company']), axis=1)
    data.drop(['CPU_Type'], axis=1, inplace=True)

    process_gpu(data)

    # Encoding Columns
    ordinal_cols = ['CPU_Family', 'CPU_Generation', 'GPU_Family', 'GPU_Series']
    ordinal_encoder = OrdinalEncoder()
    data[ordinal_cols] = ordinal_encoder.fit_transform(data[ordinal_cols].astype(str))

    # Apply One-Hot Encoding
    encoding_cols = ['Company', 'TypeName', 'CPU_Company', 'Memory Type', 'GPU_Company', 'OpSys']
    data_encoded = pd.get_dummies(data, columns=encoding_cols, drop_first=True)

    # Reorder Columns to Match Model's Expected Features
    model_features = model.feature_names_
    data_encoded = data_encoded.reindex(columns=model_features, fill_value=0)
    return data_encoded


# ---- FEATURE SELECTION TAB ----
with tabs[1]:
    st.header("🔧 Feature Selection")
    st.markdown("### Choose the specifications for the laptop:")

    user_input = {}
    col1, col2, col3 = st.columns(3)

    for idx, col in enumerate(df.columns):
        current_col = [col1, col2, col3][idx % 3]

        if col in ["Inches", "RAM (GB)"]:
            # Use selectbox for Inches and RAM (GB)
            unique_values = sorted(df[col].unique())
            user_input[col] = current_col.selectbox(f"**{col}**", unique_values)
        
        elif df[col].dtype == "object" or df[col].dtype.name == "category":
            # Use selectbox for categorical columns
            unique_values = sorted(df[col].unique())
            user_input[col] = current_col.selectbox(f"**{col}**", unique_values)
        
        else:
            # Use slider for numeric columns other than Inches and RAM (GB)
            min_val = float(df[col].min())
            max_val = float(df[col].max())
            default_val = float(df[col].median())
            user_input[col] = current_col.slider(
                f"**{col}**", 
                min_value=min_val, 
                max_value=max_val, 
                value=default_val
            )

# ---- PREDICT PRICE TAB ----
with tabs[2]:
    st.header("📊 Predict Laptop Price")
    input_data = pd.DataFrame([user_input])

    if st.button("💰 **Predict Price**"):
        try:
            preprocessed_data = preprocess_input(input_data)
            prediction = model.predict(preprocessed_data)
            st.success(f"💰 **Predicted Price: {round(prediction[0], 2)} SAR**")
            st.balloons()
        except Exception as e:
            st.error(f"**Error in Prediction:** {str(e)}")

# ---- ABOUT TAB ----
with tabs[3]:
    st.header("ℹ️ About the App")
    st.markdown(
        """
        - This app uses a **CatBoost Regressor** to predict laptop prices.
        - It is trained on a dataset containing various laptop specifications and corresponding prices.

        ## 📁 **Dataset Source**
        This project uses the **Laptelligence** dataset from [Kaggle](https://www.kaggle.com/datasets/ironwolf437/laptop-price-dataset).
        """
    )


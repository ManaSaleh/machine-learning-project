import streamlit as st
import streamlit.components.v1 as components
import joblib
import pandas as pd

# Set Page Configuration
st.set_page_config(page_title="Laptop Price Prediction & Workflow", page_icon="💻", layout="wide")

# Streamlit App Title
st.title("💻 Laptop Price Prediction")

# Tabs for Navigation
tabs = st.tabs(["🏠 Home", "🔧 Feature Selection", "📊 Predict Price", "ℹ️ About"])

# Load Model and Data
@st.cache_resource
def load_model():
    return joblib.load("../Models/cat_model-v1.pkl")

@st.cache_data
def load_test_data():
    return joblib.load("../Data/X_test.pkl")

# Initialize
model = load_model()
X_test = load_test_data()

# Home Tab
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


# Feature Selection Tab
with tabs[1]:
    st.header("🔧 Feature Selection")
    st.markdown("### Choose the specifications for the laptop:")
    
    # Organizing features into columns for better layout
    user_input = {}
    col1, col2, col3 = st.columns(3)
    for idx, col in enumerate(X_test.columns):
        current_col = [col1, col2, col3][idx % 3]

        if X_test[col].dtype == "object" or X_test[col].dtype.name == "category":
            unique_values = sorted(X_test[col].unique())
            user_input[col] = current_col.selectbox(f"**{col}**", unique_values)
        else:
            min_val = float(X_test[col].min())
            max_val = float(X_test[col].max())
            default_val = float(X_test[col].median())
            user_input[col] = current_col.slider(
                f"**{col}**", 
                min_value=min_val, 
                max_value=max_val, 
                value=default_val
            )
    st.markdown("---")

# Predict Price Tab
with tabs[2]:
    st.header("📊 Predict Laptop Price")
    st.markdown("### Review your selections and generate a price prediction.")
    
    input_data = pd.DataFrame([user_input])
    if st.button("💰 **Predict Price**"):
        try:
            prediction = model.predict(input_data)
            st.success(f"💰 **Predicted Price: {round(prediction[0], 2)} SAR**")
            st.balloons()
        except Exception as e:
            st.error(f"**Error in Prediction:** {str(e)}")

# About Tab
with tabs[3]:
    st.header("ℹ️ About the App")
    st.markdown(
        """
        ### About the Model:
        - This app uses a **CatBoost Regressor** to predict laptop prices.
        - It is trained on a dataset containing various laptop specifications and corresponding prices.
        """
    )


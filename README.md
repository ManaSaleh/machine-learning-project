# 💻 **Laptelligence: Laptop Price Prediction App**

<p align="center">
  <img src="https://img.shields.io/badge/License-MIT-blue.svg" alt="License">
  <img src="https://img.shields.io/badge/Contributors-4-brightgreen" alt="Contributors">
  <img src="https://img.shields.io/badge/Version-1.0-orange" alt="Version">
</p>

---

## 📌 **Project Overview**
**Laptelligence** is an interactive machine learning application built using **Streamlit**, designed to predict laptop prices based on specifications like processor type, RAM, GPU, and screen resolution. It leverages a pre-trained **CatBoost Regressor** model, featuring a seamless web-based interface and real-time price prediction.

---

## 🔄 **Workflow Overview**
Below is the complete workflow of how **Laptelligence** processes data, trains models, and delivers price predictions:

<object type="image/svg+xml" data="images/laptelligence_workflow.svg" width="100%">
  Your browser does not support SVG
</object>

![Overview](https://github.com/AI-bootcamp/machine-learning-project-teem-5/blob/main/images/laptelligence_workflow.svg "Laptelligence Workflow Overview")


---

## 🚀 **Key Features**
- **📊 Real-Time Predictions:** Predict laptop prices instantly using an AI-powered backend.
- **🔧 Dynamic Feature Selection:** Provide detailed specifications interactively.
- **🌐 Seamless Web Interface:** Fully responsive web app powered by Streamlit.
- **📈 Machine Learning Model:** Pre-trained **CatBoost Regressor** for high prediction accuracy.
- **💡 Clean and Interactive UI:** Intuitive tab-based navigation for better user experience.

---

## 🛠️ **Technologies Used**
### Core Tools & Frameworks
- **Programming Language:** Python
- **Web Framework:** Streamlit
- **ML Model:** CatBoost Regressor (v4 - best performing)
  
### Key Libraries
- **Data Processing:** `pandas`, `numpy`
- **Model Training:** `CatBoost`, `sklearn`
- **Data Storage:** `joblib`
- **Data Parsing:** `re`
  
---

## 📋 **Installation Guide**
Follow these steps to set up **Laptelligence** on your local environment:

### **1. Clone the Repository**
```bash
git clone https://github.com/AI-bootcamp/machine-learning-project-teem-5.git
```

### **2. Navigate to the Project Directory**
```bash
cd machine-learning-project-teem-5.git
```

### **3. Install Dependencies**
```bash
pip install -r requirements.txt
```

### **4. Run the Application**
```bash
cd Deployment
streamlit run app.py
```

---

## 📂 **Project Directory Structure**
```bash
.
├── Data                        # Datasets and processed data files
│   ├── data_cleaned_v1.csv
│   ├── dataset_encoding_v4.csv
│   ├── featured_dataset.csv
│   ├── laptop_price - dataset.csv
│   ├── processed_laptop_data.csv
│   ├── X_test_enocde.pkl
│   ├── X_test.pkl
│   └── X_test_v4.csv
├── Deployment                 # Streamlit application files
│   ├── app.py
│   └── app-v2.py
├── images                     # SVG animation and other visuals
│   └── laptelligence_workflow.svg
├── Models                    # Pre-trained models and encoders
│   ├── cat_model-v1.pkl
│   ├── cat_model-v2.cbm
│   ├── cat_model-v3.cbm
│   ├── cat_model-v4.pkl
│   ├── ordinal_encoder.pkl
│   └── training_columns.pkl
├── Notebook                  # Jupyter notebooks for training & evaluation
│   ├── 01_DataPreprocessing.ipynb
│   ├── 02_TestDataWithModels.ipynb
│   ├── 03_EnhancingModelPerformance.ipynb
│   ├── 04_Encode_Data.ipynb
│   ├── 05_ModelEnhancements.ipynb
│   ├── 06_PCA.ipynb
│   ├── 07_EnhancingModel.ipynb
│   ├── 08_model_catboost_enhancement.ipynb
│   └── 09_for_deploy.ipynb
├── README.md                 # Project documentation
└── requirements.txt          # Dependencies list
```

---

## 📈 **Model Evaluation Results**
Our performance metrics show significant improvements through iterative development and model refinement.

| **Evaluation Metric**        | **Initial Result** | **Best Result** |
|------------------------------|--------------------------|----------------------|
| **Mean Absolute Error (MAE)**| **1091.68**             | **599**              |
| **R² Score**                 | **0.66**                | **0.90**             |

---

## 🔍 **Data Preprocessing Steps**
The following steps were applied to ensure data consistency and enhance model performance:
- **Memory Extraction:** Extracted memory size and type (GB/TB).
- **Screen Resolution Processing:** Extracted width, height, IPS, and touchscreen info.
- **CPU Feature Extraction:** Extracted CPU family, generation, and manufacturer.
- **GPU Feature Engineering:** Extracted GPU series and performance tier.
- **Encoding:** Applied **Ordinal Encoding** and **One-Hot Encoding** for categorical data.

---

## 🤝 **Team Members**
| **Name**         | **Role**                       | **GitHub Profile**                |
|------------------|--------------------------------|-----------------------------------|
| **Shaden**     | ML Engineer / Model Training / Data Scientist  | [GitHub](https://github.com/shadenWq)|
| **Mana**     | ML Engineer / Model Training / Data Scientist/ UI Design | [GitHub](https://github.com/ManaSaleh)|
| **Lamees**     | ML Engineer / Model Training / Data Scientist| [GitHub](https://github.com/Lamees-F)|
| **Abdulkarim**     | ML Engineer / Model Training / Data Scientist         | [GitHub](https://github.com/IAbdulkarim5)|

---

## 📁 **Dataset Source**
This project uses [Laptop Price Dataset](https://www.kaggle.com/datasets/ironwolf437/laptop-price-dataset) from Kaggel.


---

## 📄 **License**
This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for more details.

---

## 🛡️ **Future Enhancements**
- **Model Upgrades:** Explore models like XGBoost, LightGBM, or AutoML pipelines.
- **Expanded Features:** Include more laptop features such as battery life and GPU benchmarks.
- **Data Visualizations:** Add interactive error analysis visualizations and charts.

---

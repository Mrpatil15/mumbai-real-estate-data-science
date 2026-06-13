# 🏠 Mumbai Real Estate Data Science & Valuation Model

This directory contains a complete data science and machine learning project focused on the **Mumbai housing market**. It includes a synthetic dataset generator, exploratory analysis, machine learning model training, and a premium interactive Streamlit dashboard.

## 📁 Project Directory Structure

```
real-estate-analysis/
```
Wait, the repository root is `mumbai-real-estate-data-science`. When you clone this repository, you will get a folder with that name.

## 📁 Project Directory Structure

```
mumbai-real-estate-data-science/
├── requirements.txt           ← Python dependency packages
├── data_generator.py          ← Generates synthetic Mumbai transaction records
├── mumbai_house_data.csv      ← Generated dataset (1600 records)
├── train_model.py             ← Trains Random Forest Regressor & saves pipeline
├── best_model.pkl             ← Serialized Random Forest pipeline
├── app.py                     ← Streamlit interactive valuation & analytics dashboard
├── real_estate_analysis.ipynb ← Jupyter Notebook walking through step-by-step EDA and modeling
└── README.md                  ← This documentation
```

---

## ⚡ Setup & Run Instructions (5 minutes)

Ensure you have Python 3.10+ installed.

### Step 1 — Navigate to the directory
Open a terminal and change your directory to this folder:
```bash
cd mumbai-real-estate-data-science
```

### Step 2 — (Optional) Create a Virtual Environment
It is highly recommended to isolate your packages using a virtual environment:
```bash
python -m venv venv
# On Windows (Command Prompt)
venv\Scripts\activate
# On Windows (PowerShell)
.\venv\Scripts\Activate.ps1
```

### Step 3 — Install dependencies
Install all the required data science packages:
```bash
pip install -r requirements.txt
```

### Step 4 — Generate the dataset & train the model (Already done)
If you need to regenerate the dataset or re-train the models in the future, run:
```bash
python data_generator.py
python train_model.py
```

### Step 5 — Run the Streamlit Dashboard
Launch the interactive web portal:
```bash
streamlit run app.py
```
This will automatically open your default browser to `http://localhost:8501`.

---

## 📊 Dataset Features & Variables

The generated dataset (`mumbai_house_data.csv`) simulates 1,600 residential sales transactions in Mumbai with realistic pricing correlations:

*   **Locality**: Location of the property (Bandra West, Andheri West, Powai, South Mumbai, Thane West, Navi Mumbai, Borivali West, Chembur, Worli, Juhu).
*   **Area_SqFt**: Total super built-up area in square feet.
*   **BHK**: Bedroom count (1 to 4).
*   **Bathrooms**: Number of bathrooms.
*   **Property_Age_Years**: Age of the building (0 indicates brand-new construction).
*   **Floor_Num**: Unit floor level.
*   **Total_Floors**: Total levels in the building structure.
*   **Parking_Available**: Whether reserved parking is included (0/1).
*   **Swimming_Pool**: Whether the complex has a swimming pool (0/1).
*   **Gym_Available**: Whether the complex has a gym (0/1).
*   **Lift_Available**: Whether the building contains high-speed lifts (0/1).
*   **Price_Lakhs**: Final property price in Lakhs INR (target variable).

---

## 📈 Machine Learning Model Performance

The trained model utilizes a **Random Forest Regressor** pipeline with preprocessors embedded (categorical one-hot encoders & numerical standard scalers). 

*   **R-squared ($R^2$) Score**: **~95.5%** of price variance explained.
*   **Mean Absolute Error (MAE)**: **~22.3 Lakhs** on average prediction.
*   **Main Price Drivers** (Feature Importance):
    1.  **Area (Sq.Ft)** (~48.5% impact)
    2.  **Locality: South Mumbai** (~14.9% impact)
    3.  **Locality: Bandra West** (~10.5% impact)
    4.  **Locality: Juhu** (~9.6% impact)
    5.  **Locality: Worli** (~9.4% impact)

# Vehicle-Price-Prediction
# 🚗 Vehicle Price Prediction Using Machine Learning

This project predicts **vehicle prices** based on vehicle specifications such as make, model, year, mileage, fuel type, transmission, drivetrain, body style, and more. It uses a **Random Forest Regressor** with **data imputation** built into the pipeline, making it robust to missing values.

A **Streamlit web app** is included for easy CSV-based price prediction.

---

## 📌 Features

✔ Train a vehicle price prediction model  
✔ Handles **missing values automatically** (no manual cleaning needed)  
✔ Supports CSV file upload for predictions  
✔ Download predicted results as CSV  
✔ Built using scikit-learn + Streamlit  

---

## 🧠 Model Overview

The model pipeline includes:

| Step | Technique |
|------|----------|
| Missing value handling | `SimpleImputer` (median for numeric, most_frequent for categorical) |
| Categorical Encoding | `OneHotEncoder` |
| Regression Model | `RandomForestRegressor` |
| Full Pipeline Saving | `joblib.dump()` |

---

## 📂 Project Structure

vehicle_price_prediction/
├── app.py # Streamlit web app
├── train.py # Model training script
├── requirements.txt # Dependencies
├── vehicle_price_model.joblib # Trained model (optional to include)
├── dataset_example.csv # Example input file (optional)
└── README.md


---

## ⚙ Installation & Setup

### 1️⃣ Create Virtual Environment (optional but recommended)

```bash
python -m venv .venv
source .venv/bin/activate   # Mac/Linux
.venv\Scripts\activate      # Windows
### 2️⃣ Install dependencies
pip install -r requirements.txt

### 🏋️ Training the Model

Prepare a dataset CSV containing a price column (target) and vehicle features.
python train.py --data dataset.csv

If training is successful, a model file will be saved:
vehicle_price_model.joblib

### 🌐 Running the Streamlit App
streamlit run app.py
Upload a CSV containing vehicle data (no price column required).
The app will display predictions and allow output download.
###📊 Example Columns

Your input CSV may include any of these columns (the model will automatically use the available ones):

Type	Columns
Categorical	make, model, fuel, transmission, body, exterior_color, interior_color, drivetrain, trim
Numeric	year, mileage, cylinders, doors

Missing values are handled automatically. ✔

### 🧪 Example Prediction Flow

Upload CSV via Streamlit

Model imputes missing values

Model predicts price for each row

Download results as CSV

### 📦 requirements.txt
pandas
numpy
scikit-learn
streamlit
joblib

### 🏁 Conclusion

This project demonstrates a robust machine learning pipeline for used vehicle price prediction, including:

🚀 Automated preprocessing
🔧 Data Imputation
🌐 Web-based prediction interface
📈 Realistic regression model

You are free to modify, extend, or enhance the model! Contributions welcome.





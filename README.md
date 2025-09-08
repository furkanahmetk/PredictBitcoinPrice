# 📈 Predicting Bitcoin's Next-Day Price Movement

This project demonstrates how to build a **machine learning classification model** that predicts whether Bitcoin's price will move **up (1)** or **down (0)** the following day.  

The project is structured as a guided exercise suitable for junior developers and can be completed in approximately **4 hours**.  

---

## 🎯 Goal
To create and evaluate a classification model that predicts Bitcoin’s next-day price direction using historical data.  

---

## 📦 Key Resources
- **Dataset:** [Bitcoin Historical Data on Kaggle](https://www.kaggle.com/)  
- **Libraries:** `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`, `jupyter`, `joblib`

---

## 📂 Project Structure
```
.
├── BTC-USD.csv            # Dataset (downloaded from Kaggle)
├── notebook.ipynb         # Jupyter Notebook with step-by-step workflow
├── requirements.txt       # Python dependencies
└── README.md              # Project documentation
```

---

## ⚙️ Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/bitcoin-price-prediction.git
cd bitcoin-price-prediction
```

### 2. Create and Activate a Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate       # On Linux/Mac
venv\Scripts\activate          # On Windows
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Download the Dataset
- Go to [Bitcoin Historical Data on Kaggle](https://www.kaggle.com/)  
- Download the file `BTC-USD.csv`  
- Place it in the project root directory (same location as `notebook.ipynb`)  

---

## 🚀 How to Run

1. Launch Jupyter Notebook:
   ```bash
   jupyter notebook
   ```
2. Open **`notebook.ipynb`**  
3. Follow the steps inside to:
   - Load and clean the dataset  
   - Engineer features (lags, target variable)  
   - Train Logistic Regression and Random Forest models  
   - Evaluate results with accuracy, classification report, and confusion matrix  

---

## 📊 Workflow Overview

### Phase 1: Setup and Data Loading (~30 min)
- Load `BTC-USD.csv` into a DataFrame  
- Convert `Date` column to datetime and set as index  

### Phase 2: Feature Engineering (~1 hr)
- Create **Target variable** (`1` if next-day price > today’s price, else `0`)  
- Add lag features (previous 5 closing prices)  

### Phase 3: Model Training (~1 hr)
- Train/test split (80/20, chronological order, no shuffle)  
- Train two models: **Logistic Regression** and **Random Forest**  

### Phase 4: Evaluation & Conclusion (~1.5 hrs)
- Evaluate predictions using:
  - Accuracy Score  
  - Classification Report  
  - Confusion Matrix (visualized with Seaborn)  
- Compare Logistic Regression vs Random Forest  
- Draw conclusions  

---

## 💾 Saving & Loading Models

You can save trained models to reuse them without retraining.

```python
import joblib

# Save models
joblib.dump(log_reg, "logistic_regression_model.pkl")
joblib.dump(rand_forest, "random_forest_model.pkl")

# Load models later
loaded_log_reg = joblib.load("logistic_regression_model.pkl")
loaded_rand_forest = joblib.load("random_forest_model.pkl")
```

This is useful if you want to deploy the model or run predictions without repeating the training steps.

---

## ✅ Expected Outcome
- **Random Forest** will likely outperform Logistic Regression  
- Even modest predictive accuracy (e.g., 55%) can be useful in financial forecasting  
- Insights into model bias (does it predict "up" more often than "down"?)  

---
## 📊 Conclusion

Two models were evaluated: **Logistic Regression** and **Random Forest**.  

- **Logistic Regression** achieved an accuracy of **51.35%**, with good recall for class `0` (0.87) but very poor recall for class `1` (0.14). This means it predicts the majority class (`0`) well but fails to capture minority class (`1`) effectively.  
- **Random Forest** achieved a similar accuracy of **50.17%**, with more balanced performance between classes (`0` recall = 0.53, `1` recall = 0.48). However, the overall accuracy remained close to random guessing (50%).  

The **confusion matrix** of Random Forest shows that predictions are nearly evenly split, misclassifying a large portion of both classes.  

⚠️ **Overall, both models performed poorly.** Accuracy is close to random chance, and the models struggle with class imbalance.  

---
## 📌 Notes
- This project is for **educational purposes only** and should not be used as a trading strategy.  
- The dataset and model are simplified to illustrate the ML pipeline.  

---

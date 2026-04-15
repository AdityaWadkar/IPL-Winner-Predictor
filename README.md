# 🏏 IPL Victory Predictor: 2nd Innings Win Probability Engine

[![Python](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B.svg)](https://streamlit.io/)
[![ML](https://img.shields.io/badge/ML-XGBoost%20%7C%20RandomForest-green.svg)](https://scikit-learn.org/)

## 📖 Introduction
The **IPL Victory Predictor** is a high-performance machine learning dashboard designed to forecast the outcome of IPL matches during the 2nd innings chase. Leveraging a comprehensive ball-by-ball dataset from **2008 to 2025**, the system provides dynamic, real-time win probability insights based on current match pressure, momentum, and historical performance.

---

## 🚀 Key Features
- **Real-time Sync**: Direct integration with **Cricbuzz** to fetch live scores and targets automatically.
- **Ensemble ML Engine**: A hybrid model combining **XGBoost**, **Random Forest**, and **Logistic Regression** for robust accuracy.
- **What-If Analysis**: Interactive sliders to forecast how specific game events (like losing 2 more wickets) affect the win percentage.
- **Momentum Insights**: Automated commentary on game pressure and team momentum.
- **Optimized Performance**: Instant match situation loading using a lightweight JSON-cached sampling system.

---

## 📈 Model Performance
The current production model is a **Soft-Voting Ensemble** trained on over 120,000 match situations.

### **Training Statistics**
- **Total Samples**: 120,967
- **Features**: `batting_team`, `bowling_team`, `city`, `runs_left`, `balls_left`, `wickets_remaining`, `target`, `crr`, `rrr`

### **Accuracy Results**
| Model | Test Accuracy | CV Mean (5-fold) |
| :--- | :--- | :--- |
| **XGBoost** | **91.47%** | 91.54% |
| **Random Forest** | 85.67% | 85.61% |
| **Logistic Regression** | 79.90% | 80.07% |
| **Final Ensemble** | **86.33%** | - |

### **Confusion Matrix**
| | Predicted Loss | Predicted Win |
| :--- | :--- | :--- |
| **Actual Loss** | 10,192 | 1,744 |
| **Actual Win** | 1,563 | 10,695 |

---

## 🏗️ Project Structure
```text
root/
├── app.py                # Main Entry Point (Streamlit UI)
├── notebooks/            # Data Exploration & Scraper Logic
│   ├── IPL_Predictor.ipynb
│   └── scraper.py        # Cricbuzz Sync Engine
├── scripts/              # Training & Data Prep Utility
│   ├── train_v2.py       # Robust ML Pipeline
│   └── generate_samples.py
├── data/                 # Dataset & Sample Cache
├── models/               # Serialized Ensemble Models
└── archive/              # Legacy Codebase
```

---

## 🔧 Installation & Usage

1. **Clone the Repo**:
   ```bash
   git clone https://github.com/AdityaWadkar/IPL-Winner-Predictor.git
   ```

2. **Setup Venv & Install Dependencies**:
   ```bash
   python -m venv venv
   source venv/Scripts/activate
   pip install -r requirements.txt
   ```

3. **Download the Dataset**:
   The project requires the IPL dataset. Download it from [Kaggle](https://www.kaggle.com/datasets/chaitu20/ipl-dataset2008-2025) and place the `IPL data 2008-2025.csv` file in the `data/` folder.

4. **Retrain the Model**:
   ```bash
   python scripts/train_v2.py
   ```
5. **Run the Application**:
   ```bash
   streamlit run app.py
   ```

---

## ✨ Design Concept
The interface uses a **Glassmorphism** design system with a semi-transparent, blur-heavy aesthetic to provide a sleek, premium data-analysis experience.

## ✨ Authors
- **Aditya Wadkar** - [@AdityaWadkar](https://www.github.com/AdityaWadkar)

🌟 *Don't forget to give this repository a star if you found it helpful!* 🌟

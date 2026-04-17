# 🏏 IPL Victory Predictor: Live Match Simulator

[![Python](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B.svg)](https://streamlit.io/)
[![ML](https://img.shields.io/badge/ML-XGBoost%20%7C%20RandomForest-green.svg)](https://scikit-learn.org/)
[![API](https://img.shields.io/badge/API-CricketData-orange.svg)](https://cricketdata.org/)
[![Live App](https://img.shields.io/badge/Live-Streamlit_App-red.svg)](https://iplmind.streamlit.app/)

## 📖 Introduction
The **IPL Victory Predictor** is a highly advanced, dual-engine machine learning dashboard built to forecast the trajectory and final outcome of IPL matches in **real-time**. Backed by comprehensive ball-by-ball datasets from **2008 to 2025**, the system provides predictive insight throughout both innings—calculating expected targets dynamically in the 1st Innings, and shifting gears to calculate robust Win Probabilities during the 2nd Innings chase.

---

## 🌟 Try It Live!
You can test the application live on the cloud:
👉 **[View the Live IPL Simulator App](https://ipl-winner-predictor-adityawadkar.streamlit.app/)**

---

## 🚀 Key Innovations

### **1. First-Innings Expected Score Engine**
- **Dynamic Phased Architecture**: Utilizes three completely independent regression systems optimized specifically for the phase of the game (**Powerplay**, **Middle**, and **Death Overs**).
- **Intelligent Bounding**: Analyzes venue stat-base, batting strength, and real-time momentum to output exceptionally stable **Safety-Bounded Expected Target Ranges**, backed by a dynamically calculated confidence tier based on real historical Mean Absolute Errors.

### **2. Second-Innings Win Probability Engine**
- **Ensemble Powerhouse**: Combines **XGBoost**, **Random Forest**, and **Logistic Regression** into a Soft-Voting Ensemble trained on over 120,000 unique chase scenarios. 
- **What-If Sandbox**: Run tactical sandbox analyses by dragging sliders (What if the chasing team loses 2 wickets in the next 10 balls? How does the win percentage plummet?).

### **3. Live API Synchronization**
- Forget manually entering data. The dashboard hooks up specifically with the **CricketData.org Live Match API** to fetch ongoing global scores, targets, active teams, and metadata completely instantly, automatically parsing the exact state into the machine learning pipelines.
- Implements robust error-handling protecting the UI smoothly against hit-limits or server disconnects.

### **4. Premium "Glassmorphism" UI**
- Dumps generic data styling for a polished, blur-heavy aesthetic.
- The UI mimics broadcast-tier visual fidelity including responsive Scorecard components, ordinal date formatting, and beautifully categorized model commentary.

---

## 🏗️ Project Architecture Layout
```text
root/
├── app.py                     # Main Entry Point (Streamlit UI & Routing)
├── notebooks/                 
│   ├── IPL_Predictor.ipynb
│   └── scraper.py             # CricketData Sync Engine & Limit Exception Handlers
├── scripts/                   
│   ├── train_v2.py            # Win Probability Ensemble (2nd Innings)
│   └── train_first_innings.py # Phase-Specific Target Predictor (1st Innings)
├── data/                      
│   ├── IPL data 2008-2025.csv # Core Dataset (Not tracked in git)
│   ├── h2h_recent.json        # Fast team-head-to-head cache
│   └── context_stats.json     # Lightweight historical venue lookup dictionary
├── models/                    
│   ├── ensemble_model.pkl         
│   └── first_innings_models.pkl   
└── archive/                   # Legacy Scraping Architecture
```

---

## 🔧 Installation & Local Deployment

1. **Clone the Repo**:
   ```bash
   git clone https://github.com/AdityaWadkar/IPL-Winner-Predictor.git
   cd IPL-Winner-Predictor
   ```

2. **Environment Configuration**:
   The `CricketData.org` API requires a key. 
   - Rename `.env.example` to `.env`.
   - Paste your free access key: `API_KEY=your_key_here`.

3. **Install Dependencies**:
   ```bash
   python -m venv venv
   source venv/Scripts/activate  # (On Windows use: .\venv\Scripts\activate)
   pip install -r requirements.txt
   ```

4. **Launch Application**:
   Simply spin up the Streamlit engine!
   ```bash
   streamlit run app.py
   ```

*(Note: If you wish to retrain the models from scratch, download the massive Kaggle Database mapping 2008-2025, place it in the `data/` folder, and individually trigger the generation files located inside `scripts/`.)*

---

## ✨ Authors
- **Aditya Wadkar** - [@AdityaWadkar](https://www.github.com/AdityaWadkar)

🌟 *Don't forget to give this repository a star if you found it helpful!* 🌟

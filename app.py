import streamlit as st
import pickle
import pandas as pd
import plotly.graph_objects as go
import base64
import os
import numpy as np
import re
import matplotlib.pyplot as plt
import datetime
import json
import warnings

# Suppress noisy sklearn warnings about unknown categories (expected in live data)
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')

# --- Configuration & Data ---
st.set_page_config(page_title="Cricket Match Simulator", page_icon="🎮", layout="wide")

teams = [
    "--- select ---",
    "Sunrisers Hyderabad",
    "Mumbai Indians",
    "Kolkata Knight Riders",
    "Royal Challengers Bengaluru",
    "Punjab Kings",
    "Chennai Super Kings",
    "Rajasthan Royals",
    "Delhi Capitals",
    "Gujarat Titans",
    "Lucknow Super Giants"
]

all_venues = [
    "Narendra Modi Stadium, Ahmedabad",
    "M. Chinnaswamy Stadium, Bengaluru",
    "M. A. Chidambaram Stadium, Chennai",
    "Arun Jaitley Stadium, Delhi",
    "HPCA Stadium, Dharamsala",
    "Barsapara Cricket Stadium, Guwahati",
    "Rajiv Gandhi International Stadium, Hyderabad",
    "Sawai Mansingh Stadium, Jaipur",
    "Eden Gardens, Kolkata",
    "Ekana Cricket Stadium, Lucknow",
    "Maharaja Yadavindra Singh Stadium, Mullanpur",
    "Wankhede Stadium, Mumbai",
    "Dr. Y.S. Rajasekhara Reddy Stadium, Visakhapatnam",
    "IS Bindra Stadium, Mohali",
    "MCA Stadium, Pune"
]

TEAM_MAP = {
    'Royal Challengers Bangalore': 'Royal Challengers Bengaluru',
    'Kings XI Punjab': 'Punjab Kings',
    'Delhi Daredevils': 'Delhi Capitals',
    'Deccan Chargers': 'Sunrisers Hyderabad'
}

# --- State Management ---
def initialize_state():
    default_vals = {
        'bat_team_val': teams[0],
        'bowl_team_val': teams[0],
        'target_val': 150,
        'score_val': 50,
        'overs_val': "5.0",
        'wickets_val': 1,
        'venue_val': all_venues[0],
        'predict_requested': False
    }
    for key, val in default_vals.items():
        if key not in st.session_state:
            st.session_state[key] = val
    if 'app_mode' not in st.session_state:
        st.session_state['app_mode'] = "🔵 Match Simulation"
    if 'current_sim_label' not in st.session_state:
        st.session_state['current_sim_label'] = "🔵 Manual Input / Simulation"

initialize_state()

# --- Assets ---
@st.cache_data
def get_img_as_base64(file):
    try:
        with open(file, "rb") as f:
            data = f.read()
        return base64.b64encode(data).decode()
    except:
        return ""

# Define background logic
img_data = get_img_as_base64("assets/background.jpg")
if img_data:
    bg_css = f'background: url("data:image/jpeg;base64,{img_data}") no-repeat center center fixed !important;'
else:
    bg_css = "background: linear-gradient(135deg, #09090b 0%, #1e1b4b 100%) !important;"

page_style = f"""
<style>
[data-testid="stAppViewContainer"] {{
    position: relative;
    {bg_css}
    background-size: cover !important;
}}

[data-testid="stAppViewContainer"]::before {{
    content: "";
    position: absolute;
    top: 0; left: 0; right: 0; bottom: 0;
    background: rgba(15, 23, 42, 0.55); /* 0.55 Opacity Overlay */
    z-index: 0;
}}

[data-testid="stAppViewContainer"] > div {{
    position: relative;
    z-index: 1;
}}

div[data-testid="stForm"] {{
    background: rgba(255, 255, 255, 0.05);
    backdrop-filter: blur(15px);
    border-radius: 20px;
    border: 1px solid rgba(255,255,255,0.1);
    padding: 2.5rem;
    color: white;
    box-shadow: 0 10px 40px rgba(0, 0, 0, 0.5);
}}

h1, h2, h3, h4, h5, h6, .stMarkdown p {{
    color: #ffffff;
    font-family: 'Inter', sans-serif;
}}

[data-testid="stMetricValue"] {{
    font-size: 2.2rem !important;
    font-weight: 800 !important;
    color: #fbbf24 !important;
}}

[data-testid="stMetric"] {{
    text-align: center;
    display: flex;
    flex-direction: column;
    align-items: center;
}}

.win-probability-bar {{
    height: 40px;
    width: 100%;
    border-radius: 20px;
    overflow: hidden;
    display: flex;
    margin: 20px 0;
    border: 2px solid rgba(255,255,255,0.1);
}}

.team-a-bar {{ height: 100%; transition: width 0.5s ease; }}
.team-b-bar {{ height: 100%; transition: width 0.5s ease; }}
</style>
"""

# --- Model Loading ---
class EnsembleModel:
    def __init__(self, pipes, weights=None):
        self.pipes = pipes
        self.weights = weights if weights else [1/len(pipes)] * len(pipes)
        
    def predict_proba(self, X):
        all_probas = [pipe.predict_proba(X) * w for pipe, w in zip(self.pipes, self.weights)]
        return np.sum(all_probas, axis=0)

if not os.path.exists("models/ensemble_data.pkl"):
    st.error("Model data not found! Please run train_v2.py first.")
    st.stop()

try:
    with open("models/ensemble_data.pkl", "rb") as f:
        ensemble_data = pickle.load(f)
    models_dict = ensemble_data['models']
    ensemble_model = ensemble_data['ensemble']
except Exception as e:
    st.error(f"Error loading models: {e}")
    st.stop()

def get_dataset_sample():
    try:
        if not os.path.exists('data/sample_matches.json'):
            return None
        with open('data/sample_matches.json', 'r') as f:
            samples = json.load(f)
        import random
        return random.choice(samples)
    except Exception as e:
        print(f"Error loading sample: {e}")
        return None

def get_categorized_scenarios():
    """Categorizes the JSON samples into thematic scenarios."""
    try:
        if not os.path.exists('data/sample_matches.json'): return {}
        with open('data/sample_matches.json', 'r') as f:
            samples = json.load(f)
        
        categories = {
            "🔥 Last Over Thriller": [],
            "⚠️ Middle-Order Collapse": [],
            "⛰️ Impossible Chase": [],
            "✅ Comfortable Run Chase": [],
            "🎭 Classic T20 Drama": []
        }
        
        for s in samples:
            target, score, wickets = s['target'], s['score'], s['wickets']
            ov_float = float(s['overs'])
            ov_int = int(ov_float)
            ov_dec = round((ov_float - ov_int) * 10)
            balls_bowled = (ov_int * 6) + ov_dec
            balls_left = 120 - balls_bowled
            runs_left = target - score
            rrr = (runs_left * 6) / balls_left if balls_left > 0 else 0
            
            if balls_left <= 12 and runs_left <= 25:
                categories["🔥 Last Over Thriller"].append(s)
            elif wickets >= 6 and runs_left > 50:
                categories["⚠️ Middle-Order Collapse"].append(s)
            elif rrr > 13 and balls_left > 18:
                categories["⛰️ Impossible Chase"].append(s)
            elif rrr < 7.5:
                categories["✅ Comfortable Run Chase"].append(s)
            else:
                categories["🎭 Classic T20 Drama"].append(s)
        
        # Filter out empty categories
        return {k: v for k, v in categories.items() if v}
    except: return {}

# --- Logic Functions for Buttons ---
def load_scenario_from_library(scenario_list):
    import random
    if scenario_list:
        sample = random.choice(scenario_list)
        st.session_state['bat_team_val'] = sample['batting_team']
        st.session_state['bowl_team_val'] = sample['bowling_team']
        st.session_state['target_val'] = sample['target']
        st.session_state['score_val'] = sample['score']
        st.session_state['wickets_val'] = sample['wickets']
        st.session_state['overs_val'] = sample['overs']
        
        scraped_v = sample['venue'].split(',')[0].lower()
        for v in all_venues:
            if scraped_v in v.lower():
                st.session_state['venue_val'] = v
                break
        st.session_state['predict_requested'] = False
def load_demo_data():
    st.session_state['bat_team_val'] = "Chennai Super Kings"
    st.session_state['bowl_team_val'] = "Kolkata Knight Riders"
    st.session_state['target_val'] = 185
    st.session_state['score_val'] = 120
    st.session_state['wickets_val'] = 4
    st.session_state['overs_val'] = "14.5"
    st.session_state['venue_val'] = "M. A. Chidambaram Stadium, Chennai"
    st.session_state['current_sim_label'] = "🔵 Demo Match: CSK vs KKR"
    st.session_state['predict_requested'] = False # Reset prediction on new data

def load_sample_trigger():
    sample = get_dataset_sample()
    if sample:
        st.session_state['bat_team_val'] = sample['batting_team']
        st.session_state['bowl_team_val'] = sample['bowling_team']
        st.session_state['target_val'] = sample['target']
        st.session_state['score_val'] = sample['score']
        st.session_state['wickets_val'] = sample['wickets']
        st.session_state['overs_val'] = sample['overs']
        
        # Venue match
        scraped_v = sample['venue'].split(',')[0].lower()
        for v in all_venues:
            if scraped_v in v.lower():
                st.session_state['venue_val'] = v
                break
        st.session_state['current_sim_label'] = "🎲 Random Situation from Dataset"
        st.session_state['predict_requested'] = False # Reset prediction on new data

def trigger_live_sync():
    matches = get_live_ipl_matches()
    if matches:
        second_inn = [m for m in matches if m.get('is_second_innings', False)]
        if second_inn:
            match = second_inn[0]
            teams_part = match['title'].split(',')[0].split(' vs ')
            if len(teams_part) == 2:
                raw_bat = teams_part[0].strip()
                raw_bowl = teams_part[1].strip()
                st.session_state['bat_team_val'] = TEAM_MAP.get(raw_bat, raw_bat)
                st.session_state['bowl_team_val'] = TEAM_MAP.get(raw_bowl, raw_bowl)
            
            sum_text = match['score_summary']
            if 'Target' in sum_text:
                st.session_state['target_val'] = int(re.search(r'(\d+)\s+Target', sum_text).group(1))
            
            s_match = re.search(r'(\d+)-(\d+)\s+\(([\d.]+)\)', sum_text)
            if s_match:
                st.session_state['score_val'] = int(s_match.group(1))
                st.session_state['wickets_val'] = int(s_match.group(2))
                st.session_state['overs_val'] = str(s_match.group(3))
            
            scraped_v = match.get('venue', 'Unknown').split(',')[0].lower()
            for v in all_venues:
                if scraped_v in v.lower():
                    st.session_state['venue_val'] = v
                    break
            st.session_state['current_sim_label'] = f"🟢 Live Match: {match['title']}"
            st.session_state['sync_msg'] = ("success", f"Synced Live Feed: {match['title']}")
        else:
            st.session_state['sync_msg'] = ("info", "Live match found but in 1st Innings. Results will appear during the chase.")
    else:
        st.session_state['sync_msg'] = ("warning", "No live IPL matches found on Cricbuzz currently.")
    st.session_state['predict_requested'] = False

def auto_load_scenario():
    scenarios = get_categorized_scenarios()
    cat = st.session_state.get('scenario_cat_val')
    if cat in scenarios:
        st.session_state['current_sim_label'] = f"🎯 Scenario: {cat}"
        load_scenario_from_library(scenarios[cat])

# --- Logic Modules ---
from notebooks.scraper import get_live_ipl_matches, get_mock_ipl_match

def is_ipl_season():
    if "demo" in st.query_params: return True
    now = datetime.datetime.now()
    return now.month in [3, 4, 5, 6]

@st.cache_data
def get_h2h_data(team1, team2):
    try:
        df_raw = pd.read_csv('data/IPL data 2008-2025.csv', low_memory=False)
        match_df = df_raw[['match_id', 'date', 'batting_team', 'bowling_team', 'match_won_by']].drop_duplicates('match_id')
        t_map = {'Delhi Daredevils': 'Delhi Capitals', 'Kings XI Punjab': 'Punjab Kings', 
                 'Royal Challengers Bangalore': 'Royal Challengers Bengaluru', 'Deccan Chargers': 'Sunrisers Hyderabad'}
        match_df['batting_team'] = match_df['batting_team'].replace(t_map)
        match_df['bowling_team'] = match_df['bowling_team'].replace(t_map)
        match_df['match_won_by'] = match_df['match_won_by'].replace(t_map)
        mask = ((match_df['batting_team'] == team1) & (match_df['bowling_team'] == team2)) | \
               ((match_df['batting_team'] == team2) & (match_df['bowling_team'] == team1))
        last_5 = match_df[mask].sort_values('date', ascending=False).head(5)
        return {
            'team1_wins': len(last_5[last_5['match_won_by'] == team1]),
            'team2_wins': len(last_5[last_5['match_won_by'] == team2]),
            'matches': last_5[['date', 'match_won_by']].to_dict('records')
        }
    except: return None

# --- UI Layout ---
st.markdown(page_style, unsafe_allow_html=True)

# Sidebar for Mode Selection
st.sidebar.markdown(f"""
    <div style="text-align: center; padding: 1rem; background: rgba(255,255,255,0.05); border-radius: 15px; margin-bottom: 2rem;">
        <h2 style="color: white; margin: 0;">🎮 Controller</h2>
        <p style="color: lightgray; font-size: 0.9rem;">Choose your experience</p>
    </div>
""", unsafe_allow_html=True)

app_mode = st.sidebar.radio("MODE SELECTOR", 
                           ["🔵 Match Simulation", "🟢 Live Match Sync"], 
                           key='app_mode_radio')

st.markdown("<h1 style='text-align: center; color: white;'>🏏 IPL WIN PREDICTOR</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: lightgray;'>Explore and simulate match outcomes using advanced machine learning.</p>", unsafe_allow_html=True)

# Main Logic based on Mode
if app_mode == "🟢 Live Match Sync":
    st.subheader("🟢 Live Prediction Mode")
    if is_ipl_season():
        sc1, sc2, sc3 = st.columns([1, 2, 1])
        with sc2:
            st.button("🔄 Sync Live Data from Cricbuzz", use_container_width=True, on_click=trigger_live_sync)
            
        if 'sync_msg' in st.session_state:
            m_type, txt = st.session_state['sync_msg']
            if m_type == "success": st.success(txt)
            elif m_type == "info": st.info(txt)
            elif m_type == "warning": st.warning(txt)
            del st.session_state['sync_msg']
    else:
        st.warning("IPL Season is currently inactive. Use Simulation Mode for testing!")

else:  # Match Simulation Mode
    st.subheader("🔵 Interactive Simulation Mode")
    scenarios = get_categorized_scenarios()
    
    col_a, col_b = st.columns([2, 1])
    with col_a:
        st.selectbox("🎯 Scenario Library: Select Situation Type", 
                    options=list(scenarios.keys()), 
                    key='scenario_cat_val',
                    on_change=auto_load_scenario)
    with col_b:
        st.write(" ") # Spacer
        st.write(" ")
        st.button("🎲 Generate Random Situation", use_container_width=True, on_click=load_sample_trigger)
            
    if st.session_state.get('scenario_cat_val'):
        pass # Moved below for better positioning

st.write("---")

# Active Simulation Label
sim_label = st.session_state.get('current_sim_label', "🔵 Manual Input")
st.markdown(f"""
    <div style="background: rgba(255,255,255,0.1); border-left: 5px solid #fbbf24; padding: 10px 20px; border-radius: 5px; margin-bottom: 20px;">
        <span style="color: #fbbf24; font-weight: bold; font-size: 0.9rem; text-transform: uppercase; letter-spacing: 1px;">ACTIVE CONTEXT</span><br>
        <span style="color: white; font-size: 1.2rem; font-weight: 500;">{sim_label}</span>
    </div>
""", unsafe_allow_html=True)

with st.form("prediction_form"):
    st.subheader("Match Context")
    c1, c2, c3 = st.columns(3)
    with c1: batting_team = st.selectbox("Batting Team", teams, key='bat_team_val')
    with c2: bowling_team = st.selectbox("Bowling Team", teams, key='bowl_team_val')
    with c3: selected_city = st.selectbox("Select Venue (City)", sorted(all_venues), key='venue_val')
        
    st.subheader("Current Situation")
    c4, c5 = st.columns(2)
    with c4: target = st.number_input("Target Score", min_value=1, step=1, key='target_val')
    with c5: score = st.number_input("Current Score", min_value=0, step=1, key='score_val')
        
    c6, c7 = st.columns(2)    
    with c6:
        ov_raw = st.text_input("Overs Completed (e.g. 14.2)", key='overs_val')
        try:
            overs = float(ov_raw) if ov_raw else 0.0
        except:
            overs = 0.0
    with c7: wickets_down = st.number_input("Wickets Down", min_value=0, max_value=10, step=1, key='wickets_val')
    submitted = st.form_submit_button("🔥 Predict Winning Probability", use_container_width=True)

if submitted: st.session_state['predict_requested'] = True

if st.session_state.get('predict_requested', False):
    if batting_team == "--- select ---" or bowling_team == "--- select ---" or batting_team == bowling_team:
        st.warning("Please select distinct batting and bowling teams.")
    elif score > target:
        st.warning("Score cannot exceed target.")
    elif overs > 20:
        st.warning("Overs cannot exceed 20.")
    else:
        ov_int = int(overs)
        ov_dec = round((overs - ov_int) * 10)
        balls_bowled = (ov_int * 6) + ov_dec
        balls_left = 120 - balls_bowled
        runs_left = target - score
        
        st.write("---")
        st.subheader("🛠️ Forecast: Analysis")
        adj_wickets = st.slider("Forecast: What if they lose more wickets?", 
                              min_value=wickets_down, max_value=10, value=wickets_down, key='adj_wickets_slider')
        
        crr = score / (balls_bowled / 6) if balls_bowled > 0 else 0
        rrr = runs_left / (balls_left / 6) if balls_left > 0 else 0
        
        input_data = pd.DataFrame({
            "batting_team": [batting_team], "bowling_team": [bowling_team], "city": [selected_city],
            "runs_left": [runs_left], "balls_left": [balls_left], "wickets_remaining": [10 - adj_wickets],
            "target": [target], "crr": [crr], "rrr": [rrr]
        })

        try:
            st.write("---")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Runs Required", runs_left)
            m2.metric("Balls Remaining", balls_left)
            m3.metric("Current Run Rate", f"{crr:.2f}")
            m4.metric("Required Run Rate", f"{rrr:.2f}", delta=f"{(rrr-crr):.2f}", delta_color="inverse")
            
            st.write("---")
            st.markdown(f"<h2 style='text-align:center;'>FINAL WIN PROBABILITY</h2>", unsafe_allow_html=True)
            res = ensemble_model.predict_proba(input_data)
            win_p, loss_p = res[0][1] * 100, res[0][0] * 100
            
            st.markdown(f"""
            <div style="display: flex; justify-content: space-between; font-weight: bold; font-size: 1.2rem;">
                <span>{batting_team}</span><span>{bowling_team}</span>
            </div>
            <div class="win-probability-bar">
                <div class="team-a-bar" style="width: {win_p}%; background: #10b981; display: flex; align-items: center; padding-left: 15px;">{win_p:.1f}%</div>
                <div class="team-b-bar" style="width: {loss_p}%; background: #ef4444; display: flex; align-items: center; justify-content: flex-end; padding-right: 15px;">{loss_p:.1f}%</div>
            </div>""", unsafe_allow_html=True)
            
            with st.expander("📊 Breakdown by ML Algorithm"):
                cols = st.columns(len(models_dict))
                for idx, (name, pipe) in enumerate(models_dict.items()):
                    wp = pipe.predict_proba(input_data)[0][1] * 100
                    cols[idx].write(f"**{name}**: {wp:.1f}%")
            
            st.write("---")
            ci1, ci2 = st.columns(2)
            with ci1:
                st.subheader("📈 Momentum Insight")
                if rrr - crr > 3: st.error("Pressure is intense! Momentum is dropping.")
                elif win_p > 70: st.success("Cruising with high momentum.")
                else: st.info("Evenly poised. A wicket changes everything.")
            
            with ci2:
                st.subheader("📊 Head-to-Head (Last 5)")
                h2h = get_h2h_data(batting_team, bowling_team)
                if h2h:
                    # st.write(f"Recent trend: {batting_team} ({h2h['team1_wins']}) vs {bowling_team} ({h2h['team2_wins']})")
                    # Matplotlib Horizontal Bar Chart
                    fig, ax = plt.subplots(figsize=(6, 2.5))
                    fig.patch.set_facecolor('none')
                    ax.set_facecolor('none')
                    
                    teams_h2h = [bowling_team, batting_team]
                    wins_h2h = [h2h['team2_wins'], h2h['team1_wins']]
                    colors_h2h = ['#ef4444', '#10b981']
                    
                    bars = ax.barh(teams_h2h, wins_h2h, color=colors_h2h, height=0.6)
                    ax.set_title("Head-to-Head Wins (Last 5 Encounters)", color='white', pad=15)
                    ax.tick_params(colors='white')
                    ax.spines['bottom'].set_color('white')
                    ax.spines['left'].set_color('white')
                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)
                    
                    # Add labels on bars
                    for bar, val in zip(bars, wins_h2h):
                        ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2, 
                                f'{val}', va='center', color='white', fontweight='bold')
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                    # for m in h2h['matches']: st.write(f"- {m['date']}: {m['match_won_by']}")
                else: st.caption("No H2H data found.")
        except Exception as e: st.error(f"Prediction error: {e}")

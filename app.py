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
st.set_page_config(page_title="Cricket Match Simulator", page_icon="🏏", layout="wide")

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

# Load Global First Innings Models & Context
try:
    with open('models/first_innings_models.pkl', 'rb') as f:
        first_innings_data = pickle.load(f)
        first_innings_models = first_innings_data.get('models', {})
except:
    first_innings_models = {}

try:
    with open('data/context_stats.json', 'r') as f:
        context_stats = json.load(f)
except:
    context_stats = {}

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
    if 'available_matches' not in st.session_state:
        st.session_state['available_matches'] = []
    if 'selected_match_idx' not in st.session_state:
        st.session_state['selected_match_idx'] = 0

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

.scorecard-container {{
    background: rgba(255, 255, 255, 0.05);
    backdrop-filter: blur(15px);
    border-radius: 15px;
    border: 1px solid rgba(255,255,255,0.1);
    padding: 20px;
    margin: 10px 0 25px 0;
    font-family: 'Inter', sans-serif;
}}
.scorecard-header {{
    font-size: 0.8rem;
    color: #ffb800;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 15px;
    border-bottom: 1px solid rgba(255,255,255,0.1);
    padding-bottom: 10px;
}}
.team-row {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 12px;
}}
.team-info {{
    display: flex;
    align-items: center;
    gap: 12px;
}}
.team-name {{
    font-size: 1.1rem;
    font-weight: 600;
    color: white;
}}
.score-box {{
    text-align: right;
}}
.score-main {{
    font-size: 1.2rem;
    font-weight: 800;
    color: #fbbf24;
}}
.score-overs {{
    font-size: 0.85rem;
    color: #94a3b8;
    margin-left: 5px;
}}
.match-footer {{
    margin-top: 15px;
    padding-top: 10px;
    border-top: 1px solid rgba(255,255,255,0.1);
    color: #fbbf24;
    font-weight: 600;
    font-size: 1rem;
    text-align: center;
}}
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

def get_ordinal(n):
    """Returns the ordinal suffix (st, nd, rd, th) for a number."""
    if 11 <= n % 100 <= 13:
        return "th"
    return {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")

def format_match_date(date_str):
    """Converts 'YYYY-MM-DD' to 'Day, DDth Month YYYY'."""
    try:
        dt = datetime.datetime.strptime(date_str, "%Y-%m-%d")
        day_name = dt.strftime("%a") # Thu
        day_num = dt.day
        month_name = dt.strftime("%B") # April
        year = dt.year
        return f"{day_name}, {day_num}{get_ordinal(day_num)} {month_name} {year}"
    except:
        return date_str

def render_scorecard(match):
    """Renders a premium HTML scorecard in the UI."""
    scores = match.get('scores_raw', [])
    teams = match.get('teams', [])
    
    # Process scores for both teams
    team_scores = []
    for i, team in enumerate(teams):
        # Find if this team has a score in the raw list
        score_data = next((s for s in scores if team in s.get('inning', '')), None)
        if score_data:
            r = score_data.get('r', 0)
            w = score_data.get('w', 0)
            o = score_data.get('o', 0)
            score_str = f"{r}-{w}"
            overs_str = f"({o})"
        else:
            score_str = "Yet to bat" if not match.get('match_ended') else "DNB"
            overs_str = ""
        team_scores.append({'name': team, 'score': score_str, 'overs': overs_str})

    formatted_date = format_match_date(match.get('date', ''))
    
    # Check if we should predict first innings score
    prediction_html = ""
    if not match.get('is_second_innings') and not match.get('match_ended'):
        pred_data = predict_first_innings_score(match)
        if pred_data:
            prediction_html = f"""<div style='margin-top: 15px; padding: 15px; background: rgba(0,0,0,0.2); border-radius: 10px; text-align: center; border: 1px solid rgba(255,255,255,0.05);'>
<div style='color: #fbbf24; font-size: 1.1rem; font-weight: bold;'>✅ Expected Score: {pred_data['expected_score']}</div>
<div style='color: #94a3b8; font-size: 0.85rem; margin-top: 5px;'>Confidence: {pred_data['confidence']}</div>
</div>"""
            
    html = f"""
    <div class="scorecard-container">
        <div class="scorecard-header">
            {formatted_date} • {match.get('match_num')} • {match.get('venue')}
        </div>
        <div class="team-row">
            <div class="team-info">
                <div class="team-name">{team_scores[0]['name']}</div>
            </div>
            <div class="score-box">
                <span class="score-main">{team_scores[0]['score']}</span>
                <span class="score-overs">{team_scores[0]['overs']}</span>
            </div>
        </div>
        <div class="team-row">
            <div class="team-info">
                <div class="team-name">{team_scores[1]['name']}</div>
            </div>
            <div class="score-box">
                <span class="score-main">{team_scores[1]['score']}</span>
                <span class="score-overs">{team_scores[1]['overs']}</span>
            </div>
        </div>
        <div class="match-footer">
            {match.get('status')}
        </div>
        {prediction_html}
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)

def predict_first_innings_score(match):
    """Predicts a bounded Expected Score during the First Innings using the Stacking Models."""
    if not first_innings_models or not context_stats:
        return None
        
    scores = match.get('scores_raw', [])
    teams = match.get('teams', [])
    if not teams or len(teams) < 2: return None
    
    batting_team = TEAM_MAP.get(teams[0], teams[0])
    bowling_team = TEAM_MAP.get(teams[1], teams[1])
    
    score_data = next((s for s in scores if teams[0] in s.get('inning', '')), None)
    if not score_data: return None
    
    try:
        current_runs = float(score_data.get('r', 0))
        wickets_lost = float(score_data.get('w', 0))
        overs_played = float(score_data.get('o', 0.0))
    except:
        return None
        
    if overs_played < 2.0:
        return None  # Too early for a reliable momentum prediction
        
    venue = match.get('venue', 'Unknown').split(',')[0].strip()
    
    overs_completed = overs_played
    overs_remaining = 20.0 - overs_completed
    wickets_in_hand = 10.0 - wickets_lost
    current_run_rate = current_runs / overs_completed if overs_completed > 0 else 0.0
    runs_last_3_overs = current_run_rate * min(3.0, overs_completed)
    runs_last_over = current_run_rate * min(1.0, overs_completed)
    
    venue_avg = float(context_stats.get('venue_avg', {}).get(venue, context_stats.get('global_mean', 160.0)))
    bat_avg = float(context_stats.get('bat_avg', {}).get(batting_team, context_stats.get('global_mean', 160.0)))
    bowl_avg = float(context_stats.get('bowl_avg', {}).get(bowling_team, context_stats.get('global_mean', 160.0)))
    
    batting_strength = bat_avg / venue_avg if venue_avg else 1.0
    bowling_strength = bowl_avg / venue_avg if venue_avg else 1.0
    expected_score = (bat_avg + venue_avg) / 2.0
    expected_remaining = max(0.0, expected_score - current_runs)
    pressure_index = current_run_rate * wickets_lost

    input_data = pd.DataFrame([{
        'batting_team': batting_team,
        'bowling_team': bowling_team,
        'venue': venue,
        'current_runs': current_runs,
        'overs_completed': overs_completed,
        'overs_remaining': overs_remaining,
        'wickets_lost': wickets_lost,
        'wickets_in_hand': wickets_in_hand,
        'current_run_rate': current_run_rate,
        'runs_last_3_overs': runs_last_3_overs,
        'runs_last_over': runs_last_over,
        'venue_avg': venue_avg,
        'bat_avg': bat_avg,
        'bowl_avg': bowl_avg,
        'batting_strength': batting_strength,
        'bowling_strength': bowling_strength,
        'expected_score': expected_score,
        'expected_remaining': expected_remaining,
        'pressure_index': pressure_index
    }])
    
    if overs_completed <= 6:
        model = first_innings_models.get('powerplay')
        mae = 32
        input_data = input_data.drop(columns=['runs_last_3_overs'], errors='ignore')
    elif overs_completed <= 15:
        model = first_innings_models.get('middle')
        mae = 23
    else:
        model = first_innings_models.get('death')
        mae = 6
        
    if not model: return None
    
    try:
        pred_rem = model.predict(input_data)[0]
        final_pred = int(current_runs + pred_rem)
        
        # Apply safety limits with a tighter mathematical spread 
        lower_bound = max(current_runs, final_pred - (mae / 2.0))
        upper_bound = min(300, final_pred + (mae / 2.0))
        
        if mae > 25: conf = "Low"
        elif mae >= 15: conf = "Medium"
        else: conf = "High"
        
        return {
            'expected_score': f"{int(lower_bound)} - {int(upper_bound)}",
            'confidence': conf
        }
    except Exception as e:
        print(f"Prediction Error: {e}")
        return None

def update_predictor_from_match(match):
    """Populates the prediction form fields based on a specific match object."""
    # CASE 1: MATCH COMPLETED
    if match.get('match_ended'):
        st.session_state['sync_msg'] = ("info", f"🏆 Match Completed! Check the final scorecard below.")
        st.session_state['is_chase_active'] = False
        return

    # CASE 2: 1ST INNINGS (NOT STARTED OR IN PROGRESS)
    if not match.get('is_second_innings'):
        st.session_state['sync_msg'] = ("info", "🏏 1st Innings in progress. This simulator is optimized for run chases!")
        st.session_state['is_chase_active'] = False
        # Optional: update teams anyway
        st.session_state['bat_team_val'] = TEAM_MAP.get(match['teams'][0], match['teams'][0])
        st.session_state['bowl_team_val'] = TEAM_MAP.get(match['teams'][1], match['teams'][1])
        return

    # CASE 3: 2ND INNINGS (RUN CHASE) - POPULATE INPUTS
    st.session_state['is_chase_active'] = True
    st.session_state['bat_team_val'] = TEAM_MAP.get(match.get('current_batting_team'), match.get('current_batting_team'))
    st.session_state['bowl_team_val'] = TEAM_MAP.get(match.get('current_bowling_team'), match.get('current_bowling_team'))
    
    sum_text = match['score_summary']
    if 'Target' in sum_text:
        target_match = re.search(r'(\d+)\s+Target', sum_text)
        if target_match:
            st.session_state['target_val'] = int(target_match.group(1))
    
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
    st.session_state['sync_msg'] = ("success", f"🔥 Data Synced! Target set: {st.session_state['target_val']}")

def trigger_live_sync():
    try:
        matches = get_live_ipl_matches()
        if not matches:
            st.session_state['sync_msg'] = ("warning", "No live IPL matches found for today.")
            return

        st.session_state['available_matches'] = matches
        st.session_state['selected_match_idx'] = 0 # Default to latest (first in list)
        
        match = matches[0]
        st.session_state['last_synced_match'] = match
        update_predictor_from_match(match)
        st.session_state['predict_requested'] = False
    except Exception as e:
        st.session_state['sync_msg'] = ("warning", f"API Issue: {str(e)}")

def on_match_selection_change():
    """Callback when the user selects a different match from the dropdown."""
    # Find match by title in the available list
    new_title = st.session_state['match_selector_key']
    selected_match = next((m for m in st.session_state['available_matches'] if m['title'] == new_title), None)
    
    if selected_match:
        st.session_state['last_synced_match'] = selected_match
        update_predictor_from_match(selected_match)
        st.session_state['predict_requested'] = False

def auto_load_scenario():
    scenarios = get_categorized_scenarios()
    cat = st.session_state.get('scenario_cat_val')
    if cat in scenarios:
        st.session_state['current_sim_label'] = f"🎯 Scenario: {cat}"
        load_scenario_from_library(scenarios[cat])

# --- Logic Modules ---
from notebooks.scraper import get_live_ipl_matches

def is_ipl_season():
    if "demo" in st.query_params: return True
    now = datetime.datetime.now()
    return now.month in [3, 4, 5, 6]

@st.cache_data
def get_h2h_data(team1, team2):
    if team1 == "--- select ---" or team2 == "--- select ---":
        return None
    try:
        json_path = 'data/h2h_recent.json'
        if not os.path.exists(json_path):
            return None
            
        with open(json_path, 'r') as f:
            h2h_db = json.load(f)
            
        # Get pair key (sorted alphabetically)
        t1, t2 = sorted([team1, team2])
        pair_key = f"{t1}_{t2}"
        
        matches = h2h_db.get(pair_key, [])
        if not matches:
            return None
            
        # Matches are already sorted by date desc in JSON
        last_5 = matches[:5]
        
        return {
            'team1_wins': len([m for m in last_5 if m['winner'] == team1]),
            'team2_wins': len([m for m in last_5 if m['winner'] == team2]),
            'matches': [{'date': m['date'], 'match_won_by': m['winner']} for m in last_5]
        }
    except Exception as e:
        print(f"H2H Error: {e}")
        return None

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
                           ["🟢 Live Match Sync","🔵 Match Simulation" ], 
                           key='app_mode_radio')

st.markdown("<h1 style='text-align: center; color: white;'>🏏 IPL WIN PREDICTOR</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: lightgray;'>Explore and simulate match outcomes using advanced machine learning.</p>", unsafe_allow_html=True)

# Main Logic based on Mode
if app_mode == "🟢 Live Match Sync":
    st.subheader("🟢 Live Prediction Mode")
    if is_ipl_season():
        sc1, sc2, sc3 = st.columns([1, 2, 1])
        with sc2:
            st.button("🔄 Sync Live Data from CricketAPI", use_container_width=True, on_click=trigger_live_sync)
            
        if 'sync_msg' in st.session_state:
            m_type, txt = st.session_state['sync_msg']
            if m_type == "success": st.success(txt)
            elif m_type == "info": st.info(txt)
            elif m_type == "warning": st.warning(txt)
            del st.session_state['sync_msg']

        # Removed Double Header dropdown, automatically parsing index 0 (latest)

        # Render the scorecard if match data exists
        if 'last_synced_match' in st.session_state:
            render_scorecard(st.session_state['last_synced_match'])
            if not st.session_state['last_synced_match'].get('is_second_innings') and not st.session_state['last_synced_match'].get('match_ended'):
                st.info("💡 **Note:** This predictor is specifically built for run chases. Please wait for the 2nd innings to see win probabilities!")
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

# Active Simulation Label Logic
sim_label = st.session_state.get('current_sim_label', "🔵 Manual Input")

# If we are in Live Mode but the label is still a simulation one, override it
if app_mode == "🟢 Live Match Sync" and not sim_label.startswith("🟢"):
    sim_label = "🟢 Waiting for Live Data..."
elif app_mode == "🔵 Match Simulation" and sim_label.startswith("🟢"):
    # If we switched back to simulation, reset if it was a live match
    sim_label = "🔵 Manual Input"

st.markdown(f"""
    <div style="background: rgba(255,255,255,0.1); border-left: 5px solid #fbbf24; padding: 10px 20px; border-radius: 5px; margin-bottom: 20px;">
        <span style="color: #fbbf24; font-weight: bold; font-size: 0.9rem; text-transform: uppercase; letter-spacing: 1px;">ACTIVE CONTEXT</span><br>
        <span style="color: white; font-size: 1.2rem; font-weight: 500;">{sim_label}</span>
        {"<br><span style='color: #9ca3af; font-size: 0.8rem;'>⚠️ API Delay: Data may be 1-2 overs behind real-time.</span>" if sim_label.startswith("🟢") and "Waiting" not in sim_label else ""}
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

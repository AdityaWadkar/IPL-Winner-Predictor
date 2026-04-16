import os
import requests
import re
from datetime import datetime
from dotenv import load_dotenv

def get_live_ipl_matches():
    load_dotenv()
    api_key = os.getenv('API_KEY')
    if not api_key:
        print("Error: API_KEY not found in environment.")
        return []

    url = f"https://api.cricapi.com/v1/currentMatches?apikey={api_key}&offset=0"
    
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        if data.get('status') != 'success':
            print(f"API Error: {data.get('reason', 'Unknown error')}")
            return []
            
        matches = data.get('data', [])
        ipl_matches = []
        
        from datetime import timedelta
        now = datetime.now()
        today_str = now.strftime("%Y-%m-%d")
        yesterday_str = (now - timedelta(days=1)).strftime("%Y-%m-%d")
        
        # Collect all IPL matches from today or yesterday
        candidate_matches = []
        for match in matches:
            match_name = match.get('name', '')
            match_date = match.get('date', '')
            
            if "Indian Premier League" in match_name and match_date in [today_str, yesterday_str]:
                candidate_matches.append(match)

        # Sort candidate matches by dateTimeGMT (latest first)
        candidate_matches.sort(key=lambda x: x.get('dateTimeGMT', ''), reverse=True)

        for match in candidate_matches:
                id = match.get('id')
                title = match.get('name', '')
                status = match.get('status', 'In Progress')
                venue = match.get('venue', 'Unknown')
                scores = match.get('score', [])
                teams = match.get('teams', [])
                match_ended = match.get('matchEnded', False)
                
                # Extracting Match Number (e.g., "24th Match")
                match_num_search = re.search(r'(\d+(st|nd|rd|th)\s+Match)', match_name)
                match_num = match_num_search.group(1) if match_num_search else ""
                
                is_second_innings = False
                summary = ""
                batting_team = None
                bowling_team = None
                
                # Basic parsing for existing predictor logic
                if len(scores) >= 2:
                    is_second_innings = True
                    target = scores[0].get('r', 0) + 1
                    s1 = scores[1]
                    summary = f"{s1.get('r')}-{s1.get('w')} ({s1.get('o')}) {target} Target"
                    
                    inning_name = s1.get('inning', '')
                    for t in teams:
                        if t in inning_name:
                            batting_team = t
                            bowling_team = teams[1] if teams[0] == t else teams[0]
                            break
                            
                elif len(scores) == 1:
                    s0 = scores[0]
                    summary = f"{s0.get('r')}-{s0.get('w')} ({s0.get('o')})"
                else:
                    summary = "Details Pending"
                
                match_data = {
                    "id": id,
                    "title": title,
                    "match_num": match_num,
                    "date": match_date,
                    "status": status,
                    "match_ended": match_ended,
                    "score_summary": summary,
                    "is_second_innings": is_second_innings,
                    "venue": venue,
                    "scores_raw": scores,
                    "teams": teams
                }
                
                if batting_team and bowling_team:
                    match_data['current_batting_team'] = batting_team
                    match_data['current_bowling_team'] = bowling_team
                    
                ipl_matches.append(match_data)
                
        return ipl_matches
    except Exception as e:
        print(f"Error fetching API data: {e}")
        return []

def parse_match_details(match_summary):
    details = {
        "batting_team": None,
        "bowling_team": None,
        "score": 0,
        "wickets": 0,
        "overs": 0.0,
        "target": 0
    }
    score_match = re.search(r'([A-Za-z\s]+)\s+(\d+)[-/](\d+)\s+\(([\d.]+)\)', match_summary)
    
    if score_match:
        details["batting_team_short"] = score_match.group(1).strip()
        details["score"] = int(score_match.group(2))
        details["wickets"] = int(score_match.group(3))
        details["overs"] = float(score_match.group(4))
        
    return details

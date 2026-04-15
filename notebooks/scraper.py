import requests
from bs4 import BeautifulSoup
import re

def get_live_ipl_matches():
    url = "https://www.cricbuzz.com/cricket-match/live-scores"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Matches are usually inside cb-lv-main buckets
        match_containers = soup.find_all('div', class_='cb-lv-main')
        
        ipl_matches = []
        
        for container in match_containers:
            # Check if it's an IPL match
            series_text = container.find('h2')
            if not series_text: continue
            
            s_val = series_text.text.upper()
            if "IPL" not in s_val and "INDIAN PREMIER LEAGUE" not in s_val:
                continue
            
            # Extract individual matches in this series
            matches = container.find_all('div', class_='cb-mtch-lst')
            for m in matches:
                header = m.find('h3', class_='cb-lv-scr-mtch-hdr')
                if not header: continue
                
                title = header.text.strip()
                link = "https://www.cricbuzz.com" + header.find('a')['href']
                
                # Extract score details
                score_container = m.find('div', class_='cb-scr-wll-chf')
                if not score_container: continue
                
                status_div = m.find('div', class_=re.compile(r'cb-text-(live|complete|upcoming)'))
                status_text = status_div.text.strip() if status_div else "In Progress"
                
                # Check for 2nd innings indicators
                summary = score_container.text.strip()
                is_second_innings = "Target" in summary or "needs" in summary.lower()
                
                # Extract Venue (Usually in cb-lv-scr-mtch-hdr text or sibling)
                venue = "Unknown"
                venue_div = m.find('div', class_='cb-mtch-info-stds') # Possible class for city/venue
                if not venue_div:
                    # Fallback: check if it's in the title/summary
                    venue_match = re.search(r'•\s+([^,]+),\s+([^,\n\r]+)', m.text)
                    if venue_match:
                        venue = f"{venue_match.group(2).strip()}, {venue_match.group(1).strip()}"

                match_data = {
                    "title": title,
                    "status": status_text,
                    "link": link,
                    "score_summary": summary,
                    "is_second_innings": is_second_innings,
                    "venue": venue
                }
                
                ipl_matches.append(match_data)
                
        return ipl_matches
    except Exception as e:
        print(f"Error scraping Cricbuzz: {e}")
        return []

def parse_match_details(match_summary):
    """
    Parses a string like 'CSK 87-2 (7.4) KKR' into usable dict.
    This is highly dependent on the text format.
    """
    details = {
        "batting_team": None,
        "bowling_team": None,
        "score": 0,
        "wickets": 0,
        "overs": 0.0,
        "target": 0
    }
    
    # Try to extract score information using Regex
    # Pattern: Team Name + digits-digits (digits.digits)
    # Example: "CSK 87-2 (7.4)"
    score_match = re.search(r'([A-Za-z\s]+)\s+(\d+)[-/](\d+)\s+\(([\d.]+)\)', match_summary)
    
    if score_match:
        details["batting_team_short"] = score_match.group(1).strip()
        details["score"] = int(score_match.group(2))
        details["wickets"] = int(score_match.group(3))
        details["overs"] = float(score_match.group(4))
        
    return details

# Mock data for testing when no live IPL is found
def get_mock_ipl_match():
    return [
        {
            "title": "Chennai Super Kings vs Kolkata Knight Riders, 22nd Match",
            "status": "LIVE",
            "score_summary": "CSK 148-3 (15.4) KKR 170 Target",
            "link": "#",
            "is_second_innings": True
        }
    ]

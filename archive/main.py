import base64
import streamlit as st
import pickle
import pandas as pd

st.set_page_config(page_title="IPL Victory Predictor", page_icon="🏏", layout="wide")

@st.cache_data
def get_img_as_base64(file):
    with open(file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

img = get_img_as_base64("background.jpg")

page_bg_img = f"""
<style>
/* Target the main container */
[data-testid="stAppViewContainer"] {{
    background-image: url("data:image/jpeg;base64,{img}");
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
    background-attachment: fixed;
}}

/* Target the Sidebar */
[data-testid="stSidebar"] > div:first-child {{
    background-image: url("data:image/jpeg;base64,{img}");
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
    background-attachment: fixed;
}}

</style>
"""

teams = [
    "--- select ---",
    "Sunrisers Hyderabad",
    "Mumbai Indians",
    "Kolkata Knight Riders",
    "Royal Challengers Bangalore",
    "Kings XI Punjab",
    "Chennai Super Kings",
    "Rajasthan Royals",
    "Delhi Capitals",
]

cities = [
    "Bangalore",
    "Hyderabad",
    "Kolkata",
    "Mumbai",
    "Visakhapatnam",
    "Indore",
    "Durban",
    "Chandigarh",
    "Delhi",
    "Dharamsala",
    "Ahmedabad",
    "Chennai",
    "Ranchi",
    "Nagpur",
    "Mohali",
    "Pune",
    "Bengaluru",
    "Jaipur",
    "Port Elizabeth",
    "Centurion",
    "Raipur",
    "Sharjah",
    "Cuttack",
    "Johannesburg",
    "Cape Town",
    "East London",
    "Abu Dhabi",
    "Kimberley",
    "Bloemfontein",
]

pipe = pickle.load(open("models/pipe.pkl", "rb"))

st.markdown(page_bg_img, unsafe_allow_html=True)
st.markdown("""
    # **IPL VICTORY PREDICTOR**
""")

col1, col2 = st.columns(2)
with col1:
    batting_team = st.selectbox("Select Batting Team", teams)
with col2:
    if batting_team == "--- select ---":
        bowling_team = st.selectbox("Select Bowling Team", teams)
    else:
        filtered_teams = [team for team in teams if team != batting_team]
        bowling_team = st.selectbox("Select Bowling Team", filtered_teams)

selected_city = st.selectbox("Select Venue", cities)


col1, col2 = st.columns(2)
with col1:
    target = st.number_input("Target", min_value=0, step=1)
with col2:
    score = st.number_input("Score", min_value=0, step=1)
col2, col3 = st.columns(2)    
with col2:
    overs = st.number_input("Overs Completed", min_value=0, max_value=20, step=1)
with col3:
    wickets_down = st.number_input("Wickets down", min_value=0, max_value=10, step=1)

if st.button("Predict Winning Probability"):
    if batting_team == "--- select ---":
        st.warning("Please select a batting team.")
    elif bowling_team == "--- select ---":
        st.warning("Please select a bowling team.")
    elif batting_team == bowling_team:
        st.warning("Batting and bowling team must be different.")
    elif target <= 0:
        st.warning("Please enter a valid target.")
    elif score < 0 or score > target:
        st.warning("Please enter a valid score less than or equal to the target.")
    elif overs < 0 or overs > 20:
        st.warning("Please enter a valid number of overs completed (0-20).")
    elif wickets_down < 0 or wickets_down > 10:
        st.warning("Please enter a valid number of wickets down (0-10).")
    else:
        runs_left = target - score
        balls_left = 120 - (overs * 6)
        wickets_remaining = 10 - wickets_down

        if balls_left <= 0:
            st.warning("No balls remaining. Check the overs completed.")
        else:
            crr = score / overs if overs > 0 else 0
            rrr = runs_left / (balls_left / 6)

            input_data = pd.DataFrame(
                {
                    "batting_team": [batting_team],
                    "bowling_team": [bowling_team],
                    "city": [selected_city],
                    "runs_left": [runs_left],
                    "balls_left": [balls_left],
                    "wickets_remaining": [wickets_remaining],
                    "total_runs_x": [target],
                    "crr": [crr],
                    "rrr": [rrr],
                }
            )

            result = pipe.predict_proba(input_data)
            loss = result[0][0]
            win = result[0][1]

            st.header(f"{batting_team} = {round(win * 100)}%")
            st.header(f"{bowling_team} = {round(loss * 100)}%")





















#     accuracy = []
# for i in range(101,10001):
#     x_train,x_test,y_train,y_test = tts(x,y,test_size=0.25,random_state=i)
#     trf = ColumnTransformer([
#     ('trf',OneHotEncoder(sparse=False,drop='first'),['batting_team','bowling_team','city'])
# ],
# remainder='passthrough')
#     pipe = Pipeline([
#     ('step1',trf),
#     ('step2',LogisticRegression(solver='liblinear'))
# ])
#     pipe.fit(x_train,y_train)
#     y_pred = pipe.predict(x_test)
#     accuracy.append(accuracy_score(y_test,y_pred))

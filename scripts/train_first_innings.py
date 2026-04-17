import pandas as pd
import numpy as np
import pickle
import os
import warnings

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_absolute_error

warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'IPL data 2008-2025.csv')
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'first_innings_models.pkl')


def train_phased_models():
    print("Loading data...")
    df = pd.read_csv(DATA_PATH, low_memory=False)
    df = df[df['innings'] == 1].copy()

    # Date
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df['year'] = df['date'].dt.year
    df = df.dropna(subset=['year'])

    # Team mapping
    team_mapping = {
        'Delhi Daredevils': 'Delhi Capitals',
        'Kings XI Punjab': 'Punjab Kings',
        'Royal Challengers Bangalore': 'Royal Challengers Bengaluru',
        'Deccan Chargers': 'Sunrisers Hyderabad'
    }
    df['batting_team'] = df['batting_team'].replace(team_mapping)
    df['bowling_team'] = df['bowling_team'].replace(team_mapping)

    df['venue'] = df['venue'].astype(str).apply(lambda x: x.split(',')[0].strip())

    print("Computing features...")
    df = df.sort_values(by=['match_id', 'over', 'ball'])

    # Final score
    total_scores = df.groupby('match_id')['runs_total'].sum().reset_index()
    total_scores.columns = ['match_id', 'final_score']

    df['current_runs'] = df.groupby('match_id')['runs_total'].cumsum()
    df['overs_completed'] = df['over'] + df['ball'] / 6.0
    df['overs_remaining'] = 20 - df['overs_completed']

    df['is_wicket'] = df['player_out'].notnull().astype(int)
    df['wickets_lost'] = df.groupby('match_id')['is_wicket'].cumsum()
    df['wickets_in_hand'] = 10 - df['wickets_lost']

    df['current_run_rate'] = np.where(
        df['overs_completed'] > 0,
        df['current_runs'] / df['overs_completed'],
        0
    )

    # Momentum
    df['runs_last_3_overs'] = df.groupby('match_id')['runs_total'].rolling(18, min_periods=1).sum().reset_index(0, drop=True)
    df['runs_last_over'] = df.groupby('match_id')['runs_total'].rolling(6, min_periods=1).sum().reset_index(0, drop=True)

    # Merge total
    df = df.merge(total_scores, on='match_id')

    # Target
    df['runs_remaining'] = df['final_score'] - df['current_runs']
    df['runs_remaining'] = df['runs_remaining'].clip(0, 200)

    # Context (VERY IMPORTANT)
    match_level = df.groupby('match_id').last().reset_index()
    match_level = match_level[['match_id', 'date', 'venue', 'batting_team', 'bowling_team', 'final_score']]
    match_level = match_level.sort_values('date')

    match_level['venue_avg'] = match_level.groupby('venue')['final_score'].transform(lambda x: x.expanding().mean().shift(1))
    match_level['bat_avg'] = match_level.groupby('batting_team')['final_score'].transform(lambda x: x.expanding().mean().shift(1))
    match_level['bowl_avg'] = match_level.groupby('bowling_team')['final_score'].transform(lambda x: x.expanding().mean().shift(1))

    global_mean = match_level['final_score'].mean()
    match_level[['venue_avg', 'bat_avg', 'bowl_avg']] = match_level[['venue_avg', 'bat_avg', 'bowl_avg']].fillna(global_mean)

    df = df.merge(match_level[['match_id', 'venue_avg', 'bat_avg', 'bowl_avg']], on='match_id')

    # 🔥 STRONG FEATURES
    df['batting_strength'] = df['bat_avg'] / df['venue_avg']
    df['bowling_strength'] = df['bowl_avg'] / df['venue_avg']
    df['expected_score'] = (df['bat_avg'] + df['venue_avg']) / 2
    df['expected_remaining'] = df['expected_score'] - df['current_runs']

    df['pressure_index'] = df['current_run_rate'] * df['wickets_lost']

    # Only end of over
    df = df[df['ball'] == 6]
    df = df[df['overs_completed'] >= 2.0]

    # Time split
    train_df = df[df['year'] <= 2019]
    val_df = df[(df['year'] >= 2020) & (df['year'] <= 2022)]
    test_df = df[df['year'] >= 2023]

    base_features = [
        'batting_team', 'bowling_team', 'venue',
        'current_runs', 'overs_completed', 'overs_remaining',
        'wickets_lost', 'wickets_in_hand', 'current_run_rate',
        'runs_last_3_overs', 'runs_last_over',
        'venue_avg', 'bat_avg', 'bowl_avg',
        'batting_strength', 'bowling_strength',
        'expected_score', 'expected_remaining',
        'pressure_index'
    ]

    def train_for_phase(name, min_o, max_o):
        print(f"\nTraining: {name}")

        mask_tr = (train_df['overs_completed'] > min_o) & (train_df['overs_completed'] <= max_o)
        mask_te = (test_df['overs_completed'] > min_o) & (test_df['overs_completed'] <= max_o)

        X_tr = train_df.loc[mask_tr, base_features].copy()
        y_tr = train_df.loc[mask_tr, 'runs_remaining']

        X_te = test_df.loc[mask_te, base_features].copy()

        # Remove noisy features in powerplay
        if name == 'powerplay':
            X_tr = X_tr.drop(columns=['runs_last_3_overs'])
            X_te = X_te.drop(columns=['runs_last_3_overs'])

        pre = ColumnTransformer([
            ('cat', OneHotEncoder(handle_unknown='ignore'), ['batting_team', 'bowling_team', 'venue'])
        ], remainder='passthrough')

        X_tr_t = pre.fit_transform(X_tr)
        X_te_t = pre.transform(X_te)

        # Model
        if name == 'powerplay':
            model = XGBRegressor(
                max_depth=3,
                n_estimators=250,
                learning_rate=0.03,
                subsample=0.7,
                colsample_bytree=0.7,
                reg_alpha=1,
                reg_lambda=2,
                random_state=42
            )
        else:
            model = XGBRegressor(
                max_depth=5,
                n_estimators=400,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42
            )

        model.fit(X_tr_t, y_tr)

        pred_rem = model.predict(X_te_t)
        final_pred = X_te['current_runs'].values + pred_rem

        mae = mean_absolute_error(
            test_df.loc[mask_te, 'final_score'],
            final_pred
        )

        print(f"{name} MAE: {mae:.2f}")

        return Pipeline([('pre', pre), ('model', model)]), mae

    phases = {
        'powerplay': (0, 6),
        'middle': (6, 15),
        'death': (15, 20)
    }

    models = {}
    results = {}

    for p, (mn, mx) in phases.items():
        m, score = train_for_phase(p, mn, mx)
        models[p] = m
        results[p] = score

    print("\nFINAL RESULTS")
    for k, v in results.items():
        print(f"{k}: {v:.2f}")

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

    with open(MODEL_PATH, 'wb') as f:
        pickle.dump({'models': models}, f)

    print(f"\nModels saved at: {MODEL_PATH}")


if __name__ == "__main__":
    train_phased_models()
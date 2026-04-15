import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
import os

# Define paths relative to this script
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'IPL data 2008-2025.csv')
MODEL_DIR = os.path.join(BASE_DIR, '../models')
MODEL_PATH = os.path.join(MODEL_DIR, 'ensemble_data.pkl')

warnings.filterwarnings('ignore')

print("Loading and Preprocessing dataset...")
df = pd.read_csv(DATA_PATH, low_memory=False)

# Keep only T20/IPL matches
df = df[df['innings'].isin([1, 2])]

# Rename teams to current franchises
team_mapping = {
    'Delhi Daredevils': 'Delhi Capitals',
    'Kings XI Punjab': 'Punjab Kings',
    'Royal Challengers Bangalore': 'Royal Challengers Bengaluru',
    'Deccan Chargers': 'Sunrisers Hyderabad' 
}
df['batting_team'] = df['batting_team'].replace(team_mapping)
df['bowling_team'] = df['bowling_team'].replace(team_mapping)
df['toss_winner'] = df['toss_winner'].replace(team_mapping)
df['match_won_by'] = df['match_won_by'].replace(team_mapping)

current_teams = [
    'Sunrisers Hyderabad', 'Mumbai Indians', 'Kolkata Knight Riders',
    'Royal Challengers Bengaluru', 'Punjab Kings', 'Chennai Super Kings',
    'Rajasthan Royals', 'Delhi Capitals', 'Gujarat Titans', 'Lucknow Super Giants'
]

# Filter for active teams
df = df[(df['batting_team'].isin(current_teams)) & (df['bowling_team'].isin(current_teams))]

# Calculate Total Runs for 1st Innings (The Target)
total_score_df = df[df['innings'] == 1].groupby('match_id').agg({'runs_total': 'sum'}).reset_index()
total_score_df['target'] = total_score_df['runs_total'] + 1

# Merge target with 2nd innings data
match_df = df[df['innings'] == 2].merge(total_score_df[['match_id', 'target']], on='match_id')

# Extract necessary fields
match_df['current_score'] = match_df.groupby('match_id')['runs_total'].cumsum()
match_df['runs_left'] = match_df['target'] - match_df['current_score']
match_df['runs_left'] = match_df['runs_left'].apply(lambda x: 0 if x < 0 else x)

match_df['balls_bowled'] = match_df['over'] * 6 + match_df['ball']
match_df['balls_left'] = 120 - match_df['balls_bowled']
match_df['balls_left'] = match_df['balls_left'].apply(lambda x: 0 if x < 0 else x)

match_df['player_dismissed'] = match_df['player_out'].notnull().astype(int)
match_df['wickets_falling'] = match_df.groupby('match_id')['player_dismissed'].cumsum()
match_df['wickets_remaining'] = 10 - match_df['wickets_falling']

match_df['crr'] = match_df['current_score'] * 6 / match_df['balls_bowled']
match_df['rrr'] = match_df['runs_left'] * 6 / match_df['balls_left']
match_df['rrr'] = match_df['rrr'].replace([np.inf, -np.inf], 0).fillna(0)

def result(row):
    return 1 if row['batting_team'] == row['match_won_by'] else 0

match_df['result'] = match_df.apply(result, axis=1)

final_df = match_df[['batting_team', 'bowling_team', 'city', 'runs_left', 
                      'balls_left', 'wickets_remaining', 'target', 'crr', 'rrr', 'result']]
final_df['city'] = final_df['city'].fillna(match_df['venue'])
final_df.dropna(inplace=True)

# REMOVE THIS LINE (breaks temporal ordering):
# final_df = final_df.sample(final_df.shape[0])

X = final_df.drop(columns=['result'])
y = final_df['result']

# Data exploration
print("\n--- Dataset Statistics ---")
print(f"Total samples: {len(X)}")
print(f"Features: {X.columns.tolist()}")
print(f"Class distribution:\n{y.value_counts()}")
print(f"Win rate: {y.mean():.2%}\n")

# Train/Test split with stratification
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=1,
    stratify=y  # ← ADD THIS to maintain class balance
)

print(f"Training samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")

trf = ColumnTransformer([
    ('trf', OneHotEncoder(sparse_output=False, drop='first', handle_unknown='ignore'), 
     ['batting_team', 'bowling_team', 'city'])
], remainder='passthrough')

# FIXED MODELS with regularization
models = {
    "Logistic Regression": LogisticRegression(
        solver='liblinear',
        max_iter=1000,
        random_state=1
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        max_depth=12,           # Prevent overfitting
        min_samples_split=30,
        min_samples_leaf=15,
        max_features='sqrt',
        random_state=1,
        n_jobs=-1
    ),
    "XGBoost": XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=6,
        min_child_weight=5,
        subsample=0.8,
        colsample_bytree=0.8,
        gamma=0.1,
        random_state=1,
        use_label_encoder=False,
        eval_metric='logloss'
    )
}

trained_pipes = {}
accuracy_results = {}

print("\n--- Training & Evaluating Models ---")
for name, model in models.items():
    print(f"\n{name}:")
    print("-" * 50)
    
    pipe = Pipeline([
        ('step1', trf),
        ('step2', model)
    ])
    
    # Fit on training data
    pipe.fit(X_train, y_train)
    
    # Evaluate
    train_acc = pipe.score(X_train, y_train)
    test_acc = pipe.score(X_test, y_test)
    
    # Cross-validation
    cv_scores = cross_val_score(pipe, X_train, y_train, cv=5, scoring='accuracy')
    
    trained_pipes[name] = pipe
    accuracy_results[name] = test_acc
    
    print(f"  Training Accuracy:   {train_acc:.4f}")
    print(f"  Test Accuracy:       {test_acc:.4f}")
    print(f"  CV Mean (5-fold):    {cv_scores.mean():.4f} ± {cv_scores.std()*2:.4f}")
    print(f"  Overfit Gap:         {train_acc - test_acc:.4f}")
    
    if train_acc - test_acc > 0.05:
        print("  ⚠️  WARNING: Model may be overfitting!")

# Manual Ensemble (Soft Voting)
class EnsembleModel:
    def __init__(self, pipes, weights=None):
        self.pipes = pipes
        self.weights = weights if weights else [1/len(pipes)] * len(pipes)
        
    def predict_proba(self, X):
        all_probas = [pipe.predict_proba(X) * w for pipe, w in zip(self.pipes, self.weights)]
        return np.sum(all_probas, axis=0)
    
    def predict(self, X):
        probas = self.predict_proba(X)
        return np.argmax(probas, axis=1)

print("\n" + "="*50)
print("--- Building Ensemble Model ---")
print("="*50)

ensemble = EnsembleModel(list(trained_pipes.values()))

y_pred_ensemble = ensemble.predict(X_test)
ensemble_acc = accuracy_score(y_test, y_pred_ensemble)

print(f"\nFinal Ensemble Test Accuracy: {ensemble_acc:.4f}")

# Detailed evaluation
print("\n--- Classification Report ---")
print(classification_report(y_test, y_pred_ensemble, 
                          target_names=['Loss', 'Win'], 
                          digits=4))

print("\n--- Confusion Matrix ---")
cm = confusion_matrix(y_test, y_pred_ensemble)
print(f"                Predicted Loss  Predicted Win")
print(f"Actual Loss:    {cm[0,0]:<15} {cm[0,1]}")
print(f"Actual Win:     {cm[1,0]:<15} {cm[1,1]}")

# Save everything
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

print(f"\nSaving models to {MODEL_PATH}...")
export_data = {
    'models': trained_pipes,
    'accuracy': accuracy_results,
    'ensemble': ensemble,
    'ensemble_accuracy': ensemble_acc
}

with open(MODEL_PATH, 'wb') as f:
    pickle.dump(export_data, f)

print("\n✅ Success! All models saved.")
print(f"\n📊 Summary:")
for name, acc in accuracy_results.items():
    print(f"  {name}: {acc:.4f}")
print(f"  Ensemble: {ensemble_acc:.4f}")
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import joblib

# Load dataset
df = pd.read_csv("ODI_Match_info.csv")

# Filter only major teams
valid_teams = [
    "India", "Australia", "England", "Pakistan", "South Africa",
    "New Zealand", "Sri Lanka", "Bangladesh", "Afghanistan", "West Indies"
]
df = df[df["team1"].isin(valid_teams) & df["team2"].isin(valid_teams)]
df = df[df["winner"].isin(valid_teams)]
df = df[df["toss_winner"].isin(valid_teams)]

# Remove rows with missing venue or toss_decision
df = df.dropna(subset=["venue", "toss_decision"])

# 🆕 Save Head-to-Head (overall and by venue)
h2h = df.groupby(["team1", "team2"]).agg(
    total=("winner", "count"),
    team1_wins=("winner", lambda x: (x == x.name[0]).sum())
).reset_index()
h2h["team2_wins"] = h2h["total"] - h2h["team1_wins"]
h2h.to_csv("head_to_head_stats.csv", index=False)

# 🆕 Head-to-head by venue
h2h_venue = df.groupby(["team1", "team2", "venue"]).agg(
    total=("winner", "count"),
    team1_wins=("winner", lambda x: (x == x.name[0]).sum())
).reset_index()
h2h_venue["team2_wins"] = h2h_venue["total"] - h2h_venue["team1_wins"]
h2h_venue.to_csv("head_to_head_venue_stats.csv", index=False)

# 🏟️ Venue stats
venue = df.groupby(["venue", "winner"]).size().unstack(fill_value=0)
venue["total"] = venue.sum(axis=1)
for team in valid_teams:
    venue[f"{team}_pct"] = venue.get(team, 0) / venue["total"] * 100
venue.to_csv("venue_stats.csv")

# 🔤 Label Encoding
columns = ["team1", "team2", "toss_winner", "toss_decision", "venue"]
encoders = {}
for col in columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le
    joblib.dump(le, f"encoder_{col}.pkl")

# 🆕 Encode target (winner) separately and save
le_winner = LabelEncoder()
df["winner"] = le_winner.fit_transform(df["winner"])
joblib.dump(le_winner, "encoder_winner.pkl")

# Model Training
X = df[["team1", "team2", "toss_winner", "toss_decision", "venue"]]
y = df["winner"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
joblib.dump(model, "cricket_match_predictor.pkl")

print("✅ Model trained and all files saved.")

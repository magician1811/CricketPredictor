import streamlit as st
import pandas as pd
import joblib

# Load model and encoders
model = joblib.load("cricket_match_predictor.pkl")
enc_team1 = joblib.load("encoder_team1.pkl")
enc_team2 = joblib.load("encoder_team2.pkl")
enc_toss_winner = joblib.load("encoder_toss_winner.pkl")
enc_toss_decision = joblib.load("encoder_toss_decision.pkl")
enc_venue = joblib.load("encoder_venue.pkl")
enc_winner = joblib.load("encoder_winner.pkl")

# Load match data
df = pd.read_csv("ODI_Match_info.csv")
valid_teams = list(enc_team1.classes_)
df = df[df["team1"].isin(valid_teams) & df["team2"].isin(valid_teams)]
df = df[df["winner"].isin(valid_teams)]

st.title("Cricket Match Predictor")
st.write("Select match settings to predict the winner and view stats.")

# Input options
team_list = valid_teams
venue_list = list(enc_venue.classes_)
toss_decisions = list(enc_toss_decision.classes_)

team1 = st.selectbox("Team 1", team_list)
team2 = st.selectbox("Team 2", [t for t in team_list if t != team1])
venue = st.selectbox("Venue", venue_list)
toss_winner = st.selectbox("Toss Winner", [team1, team2])
toss_decision = st.selectbox("Toss Decision", toss_decisions)

# Define current best squads
best_squads = {
    "India": ["Rohit Sharma", "Shubman Gill", "Virat Kohli", "Shreyas Iyer",
              "KL Rahul", "Hardik Pandya", "Ravindra Jadeja",
              "Kuldeep Yadav", "Jasprit Bumrah", "Mohammed Siraj", "Mohammed Shami"],
    "Pakistan": ["Babar Azam", "Fakhar Zaman", "Mohammad Rizwan", "Imam-ul-Haq",
                 "Iftikhar Ahmed", "Salman Agha", "Shadab Khan",
                 "Shaheen Afridi", "Haris Rauf", "Mohammad Wasim Jr", "Usama Mir"],
    "Australia": ["David Warner", "Travis Head", "Steve Smith", "Marnus Labuschagne",
                  "Glenn Maxwell", "Mitchell Marsh", "Josh Inglis",
                  "Pat Cummins", "Mitchell Starc", "Josh Hazlewood", "Adam Zampa"],
    "England": ["Jonny Bairstow", "Dawid Malan", "Joe Root", "Ben Stokes",
                "Jos Buttler", "Harry Brook", "Liam Livingstone",
                "Moeen Ali", "Mark Wood", "Adil Rashid", "Chris Woakes"],
    "New Zealand": ["Devon Conway", "Will Young", "Kane Williamson", "Daryl Mitchell",
                    "Glenn Phillips", "Tom Latham", "Mitchell Santner",
                    "Matt Henry", "Trent Boult", "Tim Southee", "Lockie Ferguson"],
    "South Africa": ["Quinton de Kock", "Temba Bavuma", "Rassie van der Dussen", "Aiden Markram",
                     "Heinrich Klaasen", "David Miller", "Marco Jansen",
                     "Keshav Maharaj", "Kagiso Rabada", "Lungi Ngidi", "Gerald Coetzee"],
    "Sri Lanka": ["Pathum Nissanka", "Kusal Mendis", "Charith Asalanka", "Sadeera Samarawickrama",
                  "Dasun Shanaka", "Dhananjaya de Silva", "Dunith Wellalage",
                  "Maheesh Theekshana", "Lahiru Kumara", "Dilshan Madushanka", "Matheesha Pathirana"],
    "Bangladesh": ["Litton Das", "Najmul Hossain Shanto", "Shakib Al Hasan", "Towhid Hridoy",
                   "Mushfiqur Rahim", "Mahmudullah", "Mehidy Hasan",
                   "Taskin Ahmed", "Mustafizur Rahman", "Shoriful Islam", "Nasum Ahmed"],
    "Afghanistan": ["Rahmanullah Gurbaz", "Ibrahim Zadran", "Rahmat Shah", "Hashmatullah Shahidi",
                    "Azmatullah Omarzai", "Mohammad Nabi", "Najibullah Zadran",
                    "Rashid Khan", "Mujeeb Ur Rahman", "Fazalhaq Farooqi", "Naveen-ul-Haq"]
}

# When user clicks "Predict Winner"
if st.button("Predict Winner"):
    # Encode inputs
    input_data = [[
        enc_team1.transform([team1])[0],
        enc_team2.transform([team2])[0],
        enc_toss_winner.transform([toss_winner])[0],
        enc_toss_decision.transform([toss_decision])[0],
        enc_venue.transform([venue])[0],
    ]]
    prediction = model.predict(input_data)[0]
    winner = enc_winner.inverse_transform([prediction])[0]

    st.success(f"Predicted Winner: {winner}")

    # Show current best squads
    if team1 in best_squads:
        st.subheader(f"{team1} Current Best Squad")
        team1_df = pd.DataFrame(best_squads[team1], columns=["Player"])
        st.dataframe(team1_df)
        csv1 = team1_df.to_csv(index=False).encode('utf-8')
        st.download_button(f"Download {team1} Squad", data=csv1, file_name=f"{team1}_squad.csv", mime="text/csv")

    if team2 in best_squads:
        st.subheader(f"{team2} Current Best Squad")
        team2_df = pd.DataFrame(best_squads[team2], columns=["Player"])
        st.dataframe(team2_df)
        csv2 = team2_df.to_csv(index=False).encode('utf-8')
        st.download_button(f"Download {team2} Squad", data=csv2, file_name=f"{team2}_squad.csv", mime="text/csv")

    # Head-to-Head
    h2h_df = df[((df["team1"] == team1) & (df["team2"] == team2)) |
                ((df["team1"] == team2) & (df["team2"] == team1))]
    total_matches = len(h2h_df)
    valid_h2h = h2h_df[h2h_df["winner"].isin([team1, team2])]
    t1_wins = (valid_h2h["winner"] == team1).sum()
    t2_wins = (valid_h2h["winner"] == team2).sum()
    no_result = total_matches - len(valid_h2h)

    st.subheader("Overall Head-to-Head")
    st.write(f"{team1}: {t1_wins} wins")
    st.write(f"{team2}: {t2_wins} wins")
    st.write(f"Total matches: {total_matches}")
    if no_result > 0:
        st.write(f"Draws/No Result: {no_result}")

    # Head-to-Head at venue
    h2h_venue_df = h2h_df[h2h_df["venue"] == venue]
    v_total = len(h2h_venue_df)
    valid_venue_h2h = h2h_venue_df[h2h_venue_df["winner"].isin([team1, team2])]
    v_t1_wins = (valid_venue_h2h["winner"] == team1).sum()
    v_t2_wins = (valid_venue_h2h["winner"] == team2).sum()
    v_draws = v_total - len(valid_venue_h2h)

    if v_total > 0:
        st.subheader(f"Head-to-Head at {venue}")
        st.write(f"{team1}: {v_t1_wins} wins")
        st.write(f"{team2}: {v_t2_wins} wins")
        st.write(f"Total matches at venue: {v_total}")
        if v_draws > 0:
            st.write(f"Draws/No Result at venue: {v_draws}")
    else:
        st.info("No head-to-head data available for this venue.")

    # Win rate at venue vs opponent
    t1_vs_t2_at_venue = df[(df["venue"] == venue) &
                           (((df["team1"] == team1) & (df["team2"] == team2)) |
                            ((df["team1"] == team2) & (df["team2"] == team1)))]
    v_total_specific = len(t1_vs_t2_at_venue)
    valid_specific = t1_vs_t2_at_venue[t1_vs_t2_at_venue["winner"].isin([team1, team2])]
    t1_win_vs_t2 = (valid_specific["winner"] == team1).sum()
    t2_win_vs_t1 = (valid_specific["winner"] == team2).sum()

    st.subheader("Venue Win Rate vs Opponent")
    if v_total_specific > 0 and len(valid_specific) > 0:
        t1_pct = (t1_win_vs_t2 / len(valid_specific)) * 100
        t2_pct = (t2_win_vs_t1 / len(valid_specific)) * 100
        st.write(f"{team1} at {venue} vs {team2}: {t1_pct:.1f}%")
        st.write(f"{team2} at {venue} vs {team1}: {t2_pct:.1f}%")
        if v_total_specific - len(valid_specific) > 0:
            st.write(f"Draws/No Result at venue: {v_total_specific - len(valid_specific)}")
    else:
        st.info("No win percentage data for these teams at this venue.")

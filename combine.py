import json
import pandas as pd

season = open("2015-2016.json")
skaters = pd.read_csv("Data/2015_2016/skaters.csv")
goalies = pd.read_csv("Data/2015_2016/golies.csv")
data = json.load(season)
data_len = len(data)
i = 0
for game in data:
    print(i / data_len)
    i += 1

    home = pd.DataFrame()
    for home_player in game["home_players"]:

        skater = skaters[skaters["Player"].str.find(home_player) != -1]
        if skater.size != 0:  # the player was found in the skaters info
            # Encode the positions into ints
            if skater["Pos"].iloc[0] == "LW":
                skater["Pos"].iloc[0] = 0
            elif skater["Pos"].iloc[0] == "RW":
                skater["Pos"].iloc[0] = 1
            elif skater["Pos"].iloc[0] == "C":
                skater["Pos"].iloc[0] = 2
            elif skater["Pos"].iloc[0] == "D":
                skater["Pos"].iloc[0] = 3

            home[home_player] = skater.iloc[0]
        else:  # player not found in skaters so hopefully in goalie
            goalie = goalies[goalies["Player"].str.find(home_player) != -1]
            goalie = goalie.drop(columns=["Player", "Tm", "Rk"])
            game["home_goalie_stats"] = goalie.applymap(str).to_dict('index')
    home = home.transpose()
    home = home.drop(columns=["Player", "Tm", "Rk"])  # drop redundant player name and team
    game["home_skater_stats"] = home.applymap(str).to_dict('index')

    # now for the away team stats
    away = pd.DataFrame()
    for away_player in game["away_players"]:
        skater = skaters[skaters["Player"].str.find(away_player) != -1]
        if skater.size != 0:  # the player was found in the skaters info
            # Encode the positions into ints
            if skater["Pos"].iloc[0] == "LW":
                skater["Pos"].iloc[0] = 0
            elif skater["Pos"].iloc[0] == "RW":
                skater["Pos"].iloc[0] = 1
            elif skater["Pos"].iloc[0] == "C":
                skater["Pos"].iloc[0] = 2
            elif skater["Pos"].iloc[0] == "D":
                skater["Pos"].iloc[0] = 3

            away[away_player] = skater.iloc[0]
        else:  # player not found in skaters so hopefully in goalie
            goalie = goalies[goalies["Player"].str.find(away_player) != -1]
            goalie = goalie.drop(columns=["Player", "Tm", "Rk"])
            game["away_goalie_stats"] = goalie.applymap(str).to_dict('index')
    away = away.transpose()
    away = away.drop(columns=["Player", "Tm", "Rk", "FO%"])  # drop redundant player name and team
    game["away_skater_stats"] = away.applymap(str).to_dict('index')


with open('rosters_2015-2016.json', 'w') as fout:
    json.dump(data, fout)

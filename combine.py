import json
import pandas as pd

season = open("Data/jsons/2018_2019.json")
skaters = pd.read_csv("Data/2018-2019/skaters.csv")
goalies = pd.read_csv("Data/2018-2019/goiles.csv")
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
            elif skater["Pos"].iloc[0] == "W":
                skater["Pos"].iloc[0] = 4

            home[home_player] = skater.iloc[0]
        else:  # player not found in skaters so hopefully in goalie
            goalie = goalies[goalies["Player"].str.find(home_player) != -1]
            goalie = goalie.drop(columns=["Player", "Tm", "Rk", "SV%", "QS%", "GA%-", "GSAA"])
            game["home_goalie_stats"] = goalie.applymap(str).to_dict('index')
    home = home.transpose()
    home = home.drop(columns=["Player", "Tm", "Rk", "FO%"])  # drop redundant player name and team
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
            elif skater["Pos"].iloc[0] == "W":
                skater["Pos"].iloc[0] = 4

            away[away_player] = skater.iloc[0]
        else:  # player not found in skaters so hopefully in goalie
            goalie = goalies[goalies["Player"].str.find(away_player) != -1]
            goalie = goalie.drop(columns=["Player", "Tm", "Rk", "SV%", "QS%", "GA%-", "GSAA"])
            game["away_goalie_stats"] = goalie.applymap(str).to_dict('index')
    away = away.transpose()
    away = away.drop(columns=["Player", "Tm", "Rk", "FO%"])  # drop redundant player name and team
    game["away_skater_stats"] = away.applymap(str).to_dict('index')


with open('Data/rosters/rosters_2018_2019.json', 'w') as fout:
    json.dump(data, fout)

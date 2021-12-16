from datetime import datetime
from sportsipy.nhl.boxscore import Boxscores, Boxscore
import json

# games = Boxscores(datetime(2016, 10, 12), datetime(2017, 6, 11))
games = Boxscores(datetime(2018, 10, 3), datetime(2019, 4, 6))

games_list = []
denom = len(games.games)
i = 0
for day in games.games:
    print(i / denom)
    for match in games.games[day]:
        match_dict = {}
        # Set Who the home and away teams are
        match_dict["home"] = match["home_name"]
        match_dict["away"] = match["away_name"]

        # Access Box score Values like scores and game day rosters
        box_score = Boxscore(match["boxscore"])
        match_dict["home_goals"] = box_score.home_goals
        match_dict["away_goals"] = box_score.away_goals

        # determine winner
        if box_score.winner == "Home":  # home team wins
            match_dict["outcome"] = 0
        elif box_score.winner == "Away":  # away team wins
            match_dict["outcome"] = 1
        else:
            print("Away", box_score.away_goals)
            print("Home", box_score.home_goals)
            print(box_score.winner)
            raise Exception

        # create list of home players
        home_players = []
        for player in box_score.home_players:
            home_players.append(player.name)
        match_dict["home_players"] = home_players

        # create list of away players
        away_players = []
        for player in box_score.away_players:
            away_players.append(player.name)
        match_dict["away_players"] = away_players

        # add the match to set of games
        games_list.append(match_dict)
    i += 1


print(games_list)
with open('Data/jsons/2018_2019.json', 'w') as fout:
    json.dump(games_list, fout)
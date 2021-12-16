import json

# f = open('Data/flattened_data/2015-2016_Y.json')
# data = json.load(f)
# print(data)

def convert_time_to_float(input):
    spliter = input.split(":")
    seconds = float(spliter[1]) / 60
    time_stamp = float(spliter[0]) + seconds
    return float(time_stamp)


f = open('Data/rosters/rosters_2018_2019.json')
data = json.load(f)

flattened_data_x = {}
flattened_data_y = {}
for i in range(len(data)):
    sub_flat = []

    # Add home goalie stats to the flattened list
    for home_goalie in data[i]["home_goalie_stats"]:
        for stat in data[i]["home_goalie_stats"][home_goalie]:
            sub_flat.append(float(data[i]["home_goalie_stats"][home_goalie][stat]))

    # Add home skaters stats to the flattened list
    for home_skater in data[i]["home_skater_stats"]:
        for stat in data[i]["home_skater_stats"][home_skater]:
            try:
                sub_flat.append(float(data[i]["home_skater_stats"][home_skater][stat]))
            except ValueError:
                sub_flat.append(convert_time_to_float(data[i]["home_skater_stats"][home_skater][stat]))

    # Add away goalie stats to the flattened list
    for away_goalie in data[i]["away_goalie_stats"]:
        for stat in data[i]["away_goalie_stats"][away_goalie]:
            sub_flat.append(float(data[i]["away_goalie_stats"][away_goalie][stat]))

    # Add away skaters stats to the flattened list
    for away_skater in data[i]["away_skater_stats"]:
        for stat in data[i]["away_skater_stats"][away_skater]:
            try:
                sub_flat.append(float(data[i]["away_skater_stats"][away_skater][stat]))
            except ValueError:
                # print(data[i]["away_skater_stats"][away_skater][stat])
                sub_flat.append(convert_time_to_float(data[i]["away_skater_stats"][away_skater][stat]))
    if len(sub_flat) == 902:
        flattened_data_x[str(i)] = sub_flat
        flattened_data_y[str(i)] = data[i]["outcome"]

with open('Data/flattened_data/2018-2019_X.json', 'w') as fout:
    json.dump(flattened_data_x, fout)

with open('Data/flattened_data/2018-2019_Y.json', 'w') as fout:
    json.dump(flattened_data_y, fout)
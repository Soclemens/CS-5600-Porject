import pandas as pd
import numpy as np
win_list = []
path = "Data/2019-2020/games.csv"
df = pd.read_csv(path)
for line in df.iterrows():
    '''
    Home Win = 0
    Away Win = 1
    '''
    if line[1].HG > line[1].AG:
        win_list.append(0)
    elif line[1].AG > line[1].HG:
        win_list.append(1)

# print(win_list)

df['winner'] = win_list
print(df)
df.to_csv(path)
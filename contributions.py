# pyinstaller --onefile contributions.py
from pandas import read_csv
from joblib import load
from sklearn.preprocessing import MinMaxScaler
from operator import itemgetter

ids = '01', '02', '03', '04', '05', '06', '07', '08', '09', '10'
teams = 'T1', 'T1', 'T1', 'T1', 'T1', 'T2', 'T2', 'T2', 'T2', 'T2'
roles = 'T1', 'D1', 'D2', 'S1', 'S2', 'T1', 'D1', 'D2', 'S1', 'S2'
while True:
    input('> ')
    scoreboard = read_csv('scoreboard.csv')
    death_max = load('max.joblib')
    scoreboard.loc[
        (scoreboard['E'] == 0) &
        (scoreboard['A'] == 0) &
        (scoreboard['D'] == 0) &
        (scoreboard['DMG'] == 0) &
        (scoreboard['H'] == 0) &
        (scoreboard['MIT'] == 0),
        'D'
    ] = death_max
    contributions = load('scaler.joblib').transform(scoreboard)
    contributions[:, 2] = 1 - contributions[:, 2]
    contributions *= load('importances.joblib')
    contributions = contributions.sum(1)
    tanks, damages, supports = [0, 5], [1, 2, 6, 7], [3, 4, 8, 9]
    contributions[tanks] *= 2 / contributions[tanks].sum()
    contributions[damages] *= 4 / contributions[damages].sum()
    contributions[supports] *= 4 / contributions[supports].sum()
    contributions = sorted(zip(ids, teams, roles, contributions), key=itemgetter(3), reverse=True)
    for id_, team, role, contribution in contributions:
        print(f'{id_} {team} {role} {contribution:.2f}')

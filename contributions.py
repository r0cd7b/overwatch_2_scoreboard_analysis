# pyinstaller --onefile contributions.py
import pandas as pd
import joblib
import sklearn

while True:
    scoreboard = pd.read_csv('scoreboard.csv')
    print('<Scoreboard>')
    print(scoreboard)
    death_max = joblib.load('max.joblib')
    scoreboard.loc[
        (scoreboard['E'] == 0) &
        (scoreboard['A'] == 0) &
        (scoreboard['D'] == 0) &
        (scoreboard['DMG'] == 0) &
        (scoreboard['H'] == 0) &
        (scoreboard['MIT'] == 0),
        'D'
    ] = death_max
    contributions = joblib.load('scaler.joblib').transform(scoreboard)
    contributions[:, 2] = 1 - contributions[:, 2]
    contributions *= joblib.load('importances.joblib')
    contributions = contributions.sum(1)
    tanks, damages, supports = [0, 5], [1, 2, 6, 7], [3, 4, 8, 9]
    contributions[tanks] *= 2 / contributions[tanks].sum()
    contributions[damages] *= 4 / contributions[damages].sum()
    contributions[supports] *= 4 / contributions[supports].sum()
    print('\n<Contributions>')
    print(f'TEAM 1:', end='')
    for contribution in contributions[:5]:
        print(f' {contribution:.2f}', end='')
    print(f'\nTEAM 2:', end='')
    for contribution in contributions[5:]:
        print(f' {contribution:.2f}', end='')
    input('\n> ')

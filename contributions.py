import pandas as pd
import joblib
import sklearn

while True:
    scoreboard = pd.read_csv('scoreboard.csv')
    contributions = joblib.load('scaler.joblib').transform(scoreboard)
    contributions[:, 2] = 1 - contributions[:, 2]
    contributions *= joblib.load('importances.joblib')
    contributions = contributions.sum(1)
    tanks, damages, supports = [0, 5], [1, 2, 6, 7], [3, 4, 8, 9]
    contributions[tanks] *= 2 / contributions[tanks].sum()
    contributions[damages] *= 4 / contributions[damages].sum()
    contributions[supports] *= 4 / contributions[supports].sum()
    print('<Contributions>')
    print(f'TEAM 1:', end='')
    for contribution in contributions[:5]:
        print(f' {contribution:.2f}', end='')
    print(f'\nTEAM 2:', end='')
    for contribution in contributions[5:]:
        print(f' {contribution:.2f}', end='')
    print('\n\n<Scoreboard>')
    print(scoreboard)
    input('\n> ')

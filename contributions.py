# pyinstaller -F contributions.py
try:
    import warnings
    from joblib import load
    from pandas import read_csv
    from operator import itemgetter

    warnings.simplefilter('ignore')

    import sklearn
    from numpy.core import multiarray

    death_peak = load('max.joblib')
    scaler = load('scaler.joblib')
    importances = load('importances.joblib')
    averages = importances.sum(0) / len(importances)
    while True:
        input_ = input('> ')

        scoreboard = read_csv('scoreboard.csv')
        scoreboard.loc[
            (scoreboard['E'] == 0) &
            (scoreboard['A'] == 0) &
            (scoreboard['D'] == 0) &
            (scoreboard['DMG'] == 0) &
            (scoreboard['H'] == 0) &
            (scoreboard['MIT'] == 0),
            'D'
        ] = death_peak
        scoreboard = scaler.transform(scoreboard)
        scoreboard[:, 2] = 1 - scoreboard[:, 2]
        if input_ == '':
            players = len(scoreboard)
            contributions = (scoreboard * averages).sum(1)
            contributions *= players / contributions.sum()
            contributions = sorted(zip(range(1, players + 1), contributions), key=itemgetter(1), reverse=True)
            for index, contribution in contributions:
                print(f'{index:02d} {contribution:.2f}')
            print()
        else:
            break

except Exception as e:
    input(e)

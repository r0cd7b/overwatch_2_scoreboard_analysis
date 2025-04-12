# pyinstaller -F contributions.py
try:
    import warnings
    from joblib import load
    from pandas import read_csv
    from operator import itemgetter
    import numpy.core.multiarray
    import sklearn

    warnings.simplefilter('ignore')

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
            contributions = {index: contribution for index, contribution in enumerate(contributions, 1)}
            sorted_ = sorted(contributions.items(), key=itemgetter(1), reverse=True)
            for ranking, (index, contribution) in enumerate(sorted_, 1):
                contributions[index] = contribution, ranking
            for index, (contribution, ranking) in contributions.items():
                print(f'{index:02d}: ({contribution:.2f}, {ranking:02d})')
            print()
        else:
            break
except Exception as e:
    input(e)

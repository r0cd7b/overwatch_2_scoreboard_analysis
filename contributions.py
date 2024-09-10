# pyinstaller -F contributions.py
try:
    import warnings
    from joblib import load
    from pandas import read_csv
    import numpy as np
    from operator import itemgetter

    warnings.simplefilter('ignore')

    import sklearn
    from numpy.core import multiarray

    max_d = load('max.joblib')
    scaler = load('scaler.joblib')
    role_ = load('importances.joblib')
    tank, damage, support = [0, 5], [1, 2, 6, 7], [3, 4, 8, 9]
    len_tank, len_damage, len_support = len(tank), len(damage), len(support)
    role_[tank] = role_[tank].sum(0) / len_tank
    role_[damage] = role_[damage].sum(0) / len_damage
    role_[support] = role_[support].sum(0) / len_support
    open_ = role_.sum(0) / len(role_)
    indexes = '01', '02', '03', '04', '05', '06', '07', '08', '09', '10'
    teams = 'T1', 'T1', 'T1', 'T1', 'T1', 'T2', 'T2', 'T2', 'T2', 'T2'
    roles = 'T1', 'D1', 'D2', 'S1', 'S2', 'T1', 'D1', 'D2', 'S1', 'S2'
    while True:
        print('Role: 1')
        print('Open: 2')
        print('Break: else')
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
        ] = max_d
        scoreboard = scaler.transform(scoreboard)
        scoreboard[:, 2] = 1 - scoreboard[:, 2]
        if input_ == '1':
            contributions = (scoreboard * role_).sum(1)
            contributions[tank] *= len_tank / contributions[tank].sum()
            contributions[damage] *= len_damage / contributions[damage].sum()
            contributions[support] *= len_support / contributions[support].sum()
            contributions = sorted(zip(indexes, teams, roles, contributions), key=itemgetter(3), reverse=True)
            for index, team, role, contribution in contributions:
                print(f'{index} {team} {role} {contribution:.2f}')
            print()
        elif input_ == '2':
            contributions = (scoreboard * open_).sum(1)
            contributions *= len(scoreboard) / contributions.sum()
            contributions = sorted(zip(indexes, teams, contributions), key=itemgetter(2), reverse=True)
            for index, team, contribution in contributions:
                print(f'{index} {team} {contribution:.2f}')
            print()
        else:
            break

except Exception as e:
    input(e)

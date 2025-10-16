columns = ['T', 'R', 'E1', 'E0', 'A1', 'A0', 'DE1', 'DE0', 'DA4', 'DA3', 'DA2', 'DA1', 'DA0',
           'H4', 'H3', 'H2', 'H1', 'H0', 'M4', 'M3', 'M2', 'M1', 'M0']

data_array = expanded_df[columns].to_numpy(dtype='uint8')
data_array

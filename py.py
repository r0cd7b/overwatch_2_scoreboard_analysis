from math import floor, log10

# 자리수를 분리할 열
split_columns = {'E', 'A', 'DE', 'DA', 'H', 'M'}

# 새로운 DataFrame 생성
expanded_df = DataFrame()

# 열 순회
for col in sorted_df.columns:
    if col in split_columns:
        max_val = sorted_df[col].max()
        max_digit = floor(log10(max_val))
        for digit_pos in range(max_digit, -1, -1):
            new_col_name = f'{col}{digit_pos}'
            expanded_df[new_col_name] = sorted_df[col] // 10 ** digit_pos % 10
    else:
        expanded_df[col] = sorted_df[col]
expanded_df
import pandas as pd
import numpy as np

file_path = 'E:\\resources\\work_related\\growth curve\\Backup_20230904\\all_data.xlsx'
df = pd.read_excel(file_path, header=None)
col1 = df[0]
positions = [i for i, x in enumerate(col1) if x == 'stripped_cell_line_name:']
positions
num_elements = len(positions)-2
num_list = list(range(num_elements + 1))  

df_slices = {}
df_list = []
for i in num_list:
    start = positions[i]
    end = positions[i+1] - 1
    df_slices[f'df{i+1}'] = df.iloc[start:end]
    df_list.append(df_slices[f'df{i+1}'])
    
df_slices['df1']
df_slices['df2']

df_slices[f'df{i+2}'] = df.iloc[positions[i+1]:]
df_list.append(df_slices[f'df{i+2}'])
df_list[2]

for i, df in enumerate(df_list):
    df_list[i] = df.iloc[4:]

list1 = []
list2 = []

for df in df_list:
    df.iat[0, 0] = 'Growth trends'
    value_number = df.iloc[1, 1]
    value_number = float(value_number)
    if value_number >= 200:
        list2.append(df)
    else:
        list1.append(df)

list3 = []
for df in list1:
    df = df.dropna(how='all')
    rows_to_remove = df[df[0].str.contains('Growth trends')].index.tolist()
    row = rows_to_remove[0]
    row = df.loc[row].tolist()
    row = [x for x in row if not pd.isna(x)]
    length = len(row)
    df = df.iloc[:, :length]
    list3.append(df)
list4 = []
for df in list2:
    df = df.dropna(how='all')
    rows_to_remove = df[df[0].str.contains('Growth trends')].index.tolist()
    row = rows_to_remove[0]
    row = df.loc[row].tolist()
    row = [x for x in row if not pd.isna(x)]
    length = len(row)
    df = df.iloc[:, :length]
    list4.append(df)

list5 = []
for df in list3:
    df = df.set_index(0)
    list5.append(df) 
list6 = []
for df in list4:
    df = df.set_index(0)
    list6.append(df)

list7 = []
for df in list5:
    df.columns = df.iloc[0]
    df = df.drop(df.index[0], axis=0)
    list7.append(df)
list8 = []
for df in list6:
    df.columns = df.iloc[0]
    df = df.drop(df.index[0], axis=0)
    list8.append(df)

for df in list7:    
    if "Day0" in df.columns:
        df.rename(columns={"Day0": "Day 0"}, inplace=True)
for df in list8:    
    if "Day0" in df.columns:
        df.rename(columns={"Day0": "Day 0"}, inplace=True)

merged_df = pd.concat(list7, ignore_index=False)
merged_df
merged_df = merged_df.drop(columns=['Day 8'])

merged_df.rename(columns={"Day 0": "day0"}, inplace=True)
merged_df.rename(columns={"Day 1": "day1"}, inplace=True)
merged_df.rename(columns={"Day 2": "day2"}, inplace=True)
merged_df.rename(columns={"Day 3": "day3"}, inplace=True)
merged_df.rename(columns={"Day 4": "day4"}, inplace=True)
merged_df.rename(columns={"Day 5": "day5"}, inplace=True)
merged_df.rename(columns={"Day 6": "day6"}, inplace=True)
merged_df.rename(columns={"Day 7": "day7"}, inplace=True)

merged_df = merged_df.dropna()
print(merged_df.shape[0])
for column in merged_df.columns:
    merged_df[column] = pd.to_numeric(merged_df[column], errors='coerce')
merged_df = merged_df.dropna()
print(merged_df.shape[0])
data = merged_df
print(data.isnull().any())
data = data.drop_duplicates()
len(data)


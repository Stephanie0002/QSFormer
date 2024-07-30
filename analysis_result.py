import os
import numpy as np
import pandas as pd
import re

# 创建一个空的 Pandas 数据帧来存储提取的值
data = {
    'NSS': ['rnd'] * 11 + ['hist'] * 11 + ['ind'] * 11,
    'DataSets': ['Wiki', 'UCI', 'Reddit', 'Enron', 'Mooc', 'LastFM', 'Flights', 'myket', 'SocialEvo', 'Contacts', 'Avg.Rank'] * 3,
    'JODIE': [0.0]*33,
    'DyRep': [0.0]*33,
    'TGAT': [0.0]*33,
    'TGN': [0.0]*33,
    'EdgeBank': [0.0]*33,
    'TCL': [0.0]*33,
    'GraphMixer': [0.0]*33,
    'RepeatMixer': [0.0]*33,
    'DyGFormer': [0.0]*33,
    'QSFormer': [0.0]*33,
    # 'FFNFormer': [0.0]*33
}
df = pd.read_excel('results.xlsx').rename(columns=str.lower)
out_df = pd.DataFrame(data)

neg_mapping = {'rnd':'random', 'hist':'historical', 'ind':'inductive'}
metric = 'test average_precision' # 'test average_precision' 'new node test average_precision'

for neg in ['rnd', 'hist', 'ind']:
    for model in ['JODIE', 'DyRep', 'TGAT', 'TGN', 'EdgeBank', 'TCL', 'GraphMixer', 'RepeatMixer', 'DyGFormer', 'QSFormer']:
        for dataset in ['Wiki', 'UCI', 'Reddit', 'Enron', 'Mooc', 'LastFM', 'Flights', 'myket', 'SocialEvo', 'Contacts']:
            mask_target = (out_df['NSS'] == neg) & (out_df['DataSets'] == dataset)
            mask_src = (df['model_seed'].str.lower().str.contains(model.lower())) & (df['model_seed'].str.lower().str.contains(neg_mapping[neg])) & (df['dataset'].str.lower().str.contains(dataset.lower()))
            # print(f'Processing {model} on {dataset} with {neg_mapping[neg]}', mask_src.sum())
            if(df.loc[mask_src, 'run cost(/epoch)'].values.size > 0 and df.loc[mask_src, metric].max()!=0):
                out_df.loc[mask_target, model] = df.loc[mask_src, metric].max()
            # print(out_df.loc[mask_target, model].values)
    mask_avg_rank = (out_df['NSS'] == neg) & (out_df['DataSets'] == 'Avg.Rank')
    mask_dataset_line = (out_df['NSS'] == neg) & (out_df['DataSets'] != 'Avg.Rank')
    # print(f'Processing {neg}\n', out_df.loc[mask_dataset_line, ['JODIE', 'DyRep', 'TGAT', 'TGN', 'EdgeBank', 'TCL', 'GraphMixer', 'RepeatMixer', 'DyGFormer', 'QSFormer']].rank(axis=1, method='min', ascending=False))
    out_df.iloc[np.where(mask_avg_rank)[0], 2:] = out_df.loc[mask_dataset_line, ['JODIE', 'DyRep', 'TGAT', 'TGN', 'EdgeBank', 'TCL', 'GraphMixer', 'RepeatMixer', 'DyGFormer', 'QSFormer']].rank(axis=1, method='max', ascending=False).mean(axis=0)

def percentage_format(x):
    if x==0.0:
        return 'OOT'
    if isinstance(x, (int, float)):
        return '{:.2%}'.format(x)
    else:
        return x


def decimal_format(x):
    if isinstance(x, (int, float)):
        return '{:.2f}'.format(x)
    else:
        return x
    
bad_df=out_df.copy()

# 使用 apply 函数应用这个函数
out_df.loc[out_df['DataSets'] != 'Avg.Rank'] = out_df.loc[out_df['DataSets'] != 'Avg.Rank'].applymap(percentage_format)
out_df.loc[out_df['DataSets'] == 'Avg.Rank'] = out_df.loc[out_df['DataSets'] == 'Avg.Rank'].applymap(decimal_format)

print(out_df)

# 将数据帧保存到 Excel 表格 和 LaTeX 表格
out_df.to_excel('AP_table.xlsx', index=False)
out_df.to_latex('AP_table.tex', index=False)
print('Done')


for col in bad_df.columns:
    bad_df[col] = pd.to_numeric(bad_df[col], errors='coerce')
    
# # 使用 lambda 函数和 nlargest 来找到每行的最大值和次大值
# result = bad_df.apply(lambda row: row.nlargest(2).values, axis=1)
# result = result.apply(pd.Series)
# result.columns = ['max', 'second_max']
# print(result)


# 定义一个函数，返回每行最大和次大值的列名
def get_top_two_column_names(row):
    return row.nlargest(2).index.tolist()

# 应用函数到每一行
top_two_columns = bad_df.apply(get_top_two_column_names, axis=1)

# 查看结果
print(top_two_columns)
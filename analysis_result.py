import os
import numpy as np
import pandas as pd
import re

# 创建一个空的 Pandas 数据帧来存储提取的值
data = {
    'NSS': ['rnd'] * 13 + ['hist'] * 13 + ['ind'] * 13,
    'DataSets': ['Wiki', 'UCI', 'Reddit', 'Enron', 'Mooc', 'CanParl', 'LastFM', 'Flights', 'myket', 'SocialEvo', 'Contacts', 'askUbuntu', 'Avg.Rank'] * 3,
    'JODIE': [None]*39,
    'DyRep': [None]*39,
    'TGAT': [None]*39,
    'CAWN': [None]*39,
    'EdgeBank': [None]*39,
    'TCL': [None]*39,
    'GraphMixer': [None]*39,
    'RepeatMixer': [None]*39,
    'DyGFormer': [None]*39,
    'QSFormer': [None]*39,
    'FFN-Former': [None]*39
}
df = pd.read_excel('results.xlsx').rename(columns=str.lower)
out_df = pd.DataFrame(data)

neg_mapping = {'rnd':'random', 'hist':'historical', 'ind':'inductive'}

for neg in ['rnd', 'hist', 'ind']:
    for model in ['JODIE', 'DyRep', 'TGAT', 'CAWN', 'EdgeBank', 'TCL', 'GraphMixer', 'RepeatMixer', 'DyGFormer', 'QSFormer', 'FFN-Former']:
        for dataset in ['Wiki', 'UCI', 'Reddit', 'Enron', 'Mooc', 'CanParl', 'LastFM', 'Flights', 'myket', 'SocialEvo', 'Contacts', 'askUbuntu']:
            mask_target = (out_df['NSS'] == neg) & (out_df['DataSets'] == dataset)
            mask_src = (df['model_seed'].str.lower().str.contains(model.lower())) & (df['model_seed'].str.lower().str.contains(neg_mapping[neg])) & (df['dataset'].str.lower().str.contains(dataset.lower()))
            # print(f'Processing {model} on {dataset} with {neg_mapping[neg]}', mask_src.sum())
            out_df.loc[mask_target, model] = df.loc[mask_src, 'test average_precision'].sum()
            # print(out_df.loc[mask_target, model].values)
    mask_avg_rank = (out_df['NSS'] == neg) & (out_df['DataSets'] == 'Avg.Rank')
    mask_dataset_line = (out_df['NSS'] == neg) & (out_df['DataSets'] != 'Avg.Rank')
    # print(f'Processing {neg}\n', out_df.loc[mask_dataset_line, ['JODIE', 'DyRep', 'TGAT', 'CAWN', 'EdgeBank', 'TCL', 'GraphMixer', 'RepeatMixer', 'DyGFormer', 'QSFormer', 'FFN-Former']].rank(axis=1, method='min', ascending=False))
    out_df.iloc[np.where(mask_avg_rank)[0], 2:] = out_df.loc[mask_dataset_line, ['JODIE', 'DyRep', 'TGAT', 'CAWN', 'EdgeBank', 'TCL', 'GraphMixer', 'RepeatMixer', 'DyGFormer', 'QSFormer', 'FFN-Former']].rank(axis=1, method='min', ascending=False).mean(axis=0)

def percentage_format(x):
    if isinstance(x, (int, float)):
        return '{:.2%}'.format(x)
    else:
        return x


def decimal_format(x):
    if isinstance(x, (int, float)):
        return '{:.2f}'.format(x)
    else:
        return x

# 使用 apply 函数应用这个函数
out_df.loc[out_df['DataSets'] != 'Avg.Rank'] = out_df.loc[out_df['DataSets'] != 'Avg.Rank'].applymap(percentage_format)
out_df.loc[out_df['DataSets'] == 'Avg.Rank'] = out_df.loc[out_df['DataSets'] == 'Avg.Rank'].applymap(decimal_format)

# 将数据帧保存到 Excel 表格 和 LaTeX 表格
out_df.to_excel('AP_table.xlsx', index=False)
out_df.to_latex('AP_table.tex', index=False)
print('Done')

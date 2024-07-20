import os
import numpy as np
import pandas as pd
import re

# 创建一个空的 Pandas 数据帧来存储提取的值
data = {
    'NSS': ['rnd'] * 12 + ['hist'] * 12 + ['ind'] * 12,
    'DataSets': ['Wiki', 'UCI', 'Reddit', 'Enron', 'Mooc', 'LastFM', 'Flights', 'myket', 'SocialEvo', 'Contacts', 'askUbuntu', 'Avg.Rank'] * 3,
    'JODIE': [None]*36,
    'DyRep': [None]*36,
    'TGAT': [None]*36,
    'TGN': [None]*36,
    'EdgeBank': [None]*36,
    'TCL': [None]*36,
    'GraphMixer': [None]*36,
    'RepeatMixer': [None]*36,
    'DyGFormer': [None]*36,
    'QSFormer': [None]*36,
    'FFNFormer': [None]*36
}
df = pd.read_excel('results.xlsx').rename(columns=str.lower)
out_df = pd.DataFrame(data)

neg_mapping = {'rnd':'random', 'hist':'historical', 'ind':'inductive'}

for neg in ['rnd', 'hist', 'ind']:
    for model in ['JODIE', 'DyRep', 'TGAT', 'TGN', 'EdgeBank', 'TCL', 'GraphMixer', 'RepeatMixer', 'DyGFormer', 'QSFormer', 'FFNFormer']:
        for dataset in ['Wiki', 'UCI', 'Reddit', 'Enron', 'Mooc', 'LastFM', 'Flights', 'myket', 'SocialEvo', 'Contacts', 'askUbuntu']:
            mask_target = (out_df['NSS'] == neg) & (out_df['DataSets'] == dataset)
            mask_src = (df['model_seed'].str.lower().str.contains(model.lower())) & (df['model_seed'].str.lower().str.contains(neg_mapping[neg])) & (df['dataset'].str.lower().str.contains(dataset.lower()))
            # print(f'Processing {model} on {dataset} with {neg_mapping[neg]}', mask_src.sum())
            if(df.loc[mask_src, 'run cost(/epoch)'].values.size > 0):
                out_df.loc[mask_target, model] = df.loc[mask_src, 'test average_precision'].sum()
            # print(out_df.loc[mask_target, model].values)
    mask_avg_rank = (out_df['NSS'] == neg) & (out_df['DataSets'] == 'Avg.Rank')
    mask_dataset_line = (out_df['NSS'] == neg) & (out_df['DataSets'] != 'Avg.Rank')
    # print(f'Processing {neg}\n', out_df.loc[mask_dataset_line, ['JODIE', 'DyRep', 'TGAT', 'TGN', 'EdgeBank', 'TCL', 'GraphMixer', 'RepeatMixer', 'DyGFormer', 'QSFormer', 'FFNFormer']].rank(axis=1, method='min', ascending=False))
    out_df.iloc[np.where(mask_avg_rank)[0], 2:] = out_df.loc[mask_dataset_line, ['JODIE', 'DyRep', 'TGAT', 'TGN', 'EdgeBank', 'TCL', 'GraphMixer', 'RepeatMixer', 'DyGFormer', 'QSFormer', 'FFNFormer']].rank(axis=1, method='min', ascending=False).mean(axis=0)

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

print(out_df)

# 将数据帧保存到 Excel 表格 和 LaTeX 表格
out_df.to_excel('AP_table.xlsx', index=False)
out_df.to_latex('AP_table.tex', index=False)
print('Done')

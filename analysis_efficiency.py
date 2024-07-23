import os
import numpy as np
import pandas as pd
import re

# 创建一个空的 Pandas 数据帧来存储提取的值
data = {
    'DataSets': ['Wiki', 'UCI', 'Reddit', 'Enron', 'Mooc', 'LastFM', 'Flights', 'myket', 'SocialEvo', 'Contacts', 'Avg.Rank'],
    'JODIE': [1e6]*11,
    'DyRep': [1e6]*11,
    'TGAT': [1e6]*11,
    'TGN': [1e6]*11,
    'TCL': [1e6]*11,
    'GraphMixer': [1e6]*11,
    'RepeatMixer': [1e6]*11,
    'DyGFormer': [1e6]*11,
    'QSFormer': [1e6]*11,
    # 'FFNFormer': [1e6]*11
}
df = pd.read_excel('results_e.xlsx').rename(columns=str.lower)
out_df = pd.DataFrame(data)

for model in ['JODIE', 'DyRep', 'TGAT', 'TGN', 'TCL', 'GraphMixer', 'RepeatMixer', 'DyGFormer', 'QSFormer']:
    for dataset in ['Wiki', 'UCI', 'Reddit', 'Enron', 'Mooc', 'LastFM', 'Flights', 'myket', 'SocialEvo', 'Contacts']:
        mask_target = (out_df['DataSets'] == dataset)
        mask_src = (df['model_seed'].str.lower().str.contains(model.lower())) & (df['dataset'].str.lower().str.contains(dataset.lower()))
        # print(f'Processing {model} on {dataset} with {neg_mapping[neg]}', mask_src.sum())
        if(df.loc[mask_src, 'run cost(/epoch)'].values.size > 0):
            out_df.loc[mask_target, model] = df.loc[mask_src, 'train time'].sum()
        # print(out_df.loc[mask_target, model].values)
mask_avg_rank = (out_df['DataSets'] == 'Avg.Rank')
mask_dataset_line = (out_df['DataSets'] != 'Avg.Rank')
# print(f'Processing {neg}\n', out_df.loc[mask_dataset_line, ['JODIE', 'DyRep', 'TGAT', 'TGN', 'TCL', 'GraphMixer', 'RepeatMixer', 'DyGFormer', 'QSFormer']].rank(axis=1, method='min', ascending=False))
out_df.iloc[np.where(mask_avg_rank)[0], 1:] = out_df.loc[mask_dataset_line, ['JODIE', 'DyRep', 'TGAT', 'TGN', 'TCL', 'GraphMixer', 'RepeatMixer', 'DyGFormer', 'QSFormer']].rank(axis=1, method='max', ascending=True).mean(axis=0)

def percentage_format(x):
    if isinstance(x, (int, float)):
        return '{:.2%}'.format(x)
    else:
        return x


def decimal_format(x):
    if x==1e6:
        return 'OOT'
    if isinstance(x, (int, float)):
        return '{:.2f}'.format(x)
    else:
        return x

# 使用 apply 函数应用这个函数
out_df.loc[out_df['DataSets'] != 'Avg.Rank'] = out_df.loc[out_df['DataSets'] != 'Avg.Rank'].applymap(decimal_format)
out_df.loc[out_df['DataSets'] == 'Avg.Rank'] = out_df.loc[out_df['DataSets'] == 'Avg.Rank'].applymap(decimal_format)

print(out_df)

# 将数据帧保存到 Excel 表格 和 LaTeX 表格
out_df.to_excel('E_table.xlsx', index=False)
out_df.to_latex('E_table.tex', index=False)
print('Done')

import os
import numpy as np
import pandas as pd
import re

# 创建一个空的 Pandas 数据帧来存储提取的值
data = {
    'Model': ['DyGFormer', 'QSFormer', 'Speedup',],
    'Wiki': [1e6]*3,
    'UCI': [1e6]*3,
    'Reddit': [1e6]*3,
    'Enron': [1e6]*3,
    'Mooc': [1e6]*3,
    'LastFM': [1e6]*3,
    'Flights': [1e6]*3,
    'myket': [1e6]*3,
    'SocialEvo': [1e6]*3,
    'Contacts': [1e6]*3,
    'Avg': [1e6]*3
}
df = pd.read_excel('results_e.xlsx').rename(columns=str.lower)
out_df = pd.DataFrame(data)

for model in ['DyGFormer', 'QSFormer']:
    for dataset in ['Wiki', 'UCI', 'Reddit', 'Enron', 'Mooc', 'LastFM', 'Flights', 'myket', 'SocialEvo', 'Contacts']:
        mask_target = (out_df['Model'] == model)
        mask_src = (df['file_path'].str.lower().str.contains(model.lower())) & (df['dataset'].str.lower().str.contains(dataset.lower())) & (df['file_path'].str.lower().str.contains('.log.same.effi'))
        # print(f'Processing {model} on {dataset} with {neg_mapping[neg]}', mask_src.sum())
        if(df.loc[mask_src, 'run cost(/epoch)'].values.size > 0):
            out_df.loc[mask_target, dataset] = df.loc[mask_src, 'neighbor sample time'].min()
        # print(out_df.loc[mask_target, model].values)
# compute speedup
mask_speedup = (out_df['Model'] == 'Speedup')
mask_model_line = (out_df['Model'] != 'Speedup')
out_df.iloc[np.where(mask_speedup)[0], 1:11] = out_df.loc[out_df['Model'] == 'DyGFormer', ['Wiki', 'UCI', 'Reddit', 'Enron', 'Mooc', 'LastFM', 'Flights', 'myket', 'SocialEvo', 'Contacts']].reset_index(drop=True) / out_df.loc[out_df['Model'] == 'QSFormer', ['Wiki', 'UCI', 'Reddit', 'Enron', 'Mooc', 'LastFM', 'Flights', 'myket', 'SocialEvo', 'Contacts']].reset_index(drop=True)
# compute average
out_df.loc[mask_speedup, 'Avg'] = out_df.loc[mask_speedup, ['Wiki', 'UCI', 'Reddit', 'Enron', 'Mooc', 'LastFM', 'Flights', 'myket', 'SocialEvo', 'Contacts']].mean(axis=1)
out_df.loc[out_df['Model'] == 'QSFormer', 'Avg'] = out_df.loc[out_df['Model'] == 'QSFormer', ['Wiki', 'UCI', 'Reddit', 'Enron', 'Mooc', 'LastFM', 'Flights', 'myket', 'SocialEvo', 'Contacts']].mean(axis=1)
out_df.loc[out_df['Model'] == 'DyGFormer', 'Avg'] = out_df.loc[out_df['Model'] == 'DyGFormer', ['Wiki', 'UCI', 'Reddit', 'Enron', 'Mooc', 'LastFM', 'Flights', 'myket', 'SocialEvo', 'Contacts']].mean(axis=1)

def percentage_format(x):
    if isinstance(x, (int, float)):
        return '{:.2%}'.format(x)
    else:
        return x


def decimal_format_4(x):
    if x==1e6:
        return 'OOT'
    if isinstance(x, (int, float)):
        return '{:.4f}'.format(x)
    else:
        return x
    
def interger_format(x):
    if x==1e6:
        return 'OOT'
    if isinstance(x, (int, float)):
        return '{:.0f}'.format(x)
    else:
        return x

# 使用 apply 函数应用这个函数
out_df.loc[out_df['Model'] != 'Speedup'] = out_df.loc[out_df['Model'] != 'Speedup'].applymap(decimal_format_4)
out_df.loc[out_df['Model'] == 'Speedup'] = out_df.loc[out_df['Model'] == 'Speedup'].applymap(interger_format)

print(out_df)

# 将数据帧保存到 Excel 表格 和 LaTeX 表格
out_df.to_excel('SampleE_table.xlsx', index=False)
out_df.to_latex('SampleE_table.tex', index=False)
print('Done')

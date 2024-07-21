import os
import pandas as pd
import re

# 创建一个空的 Pandas 数据帧来存储提取的值
data = {
    'file_path': [],
    'file_name': [],
    'dataset': [],
    'model_seed': [],
    'validate average_precision': [],
    'validate roc_auc': [],
    'validate mrr': [],
    'new node validate average_precision': [],
    'new node validate roc_auc': [],
    'new node validate mrr': [],
    'test average_precision': [],
    'test roc_auc': [],
    'test mrr': [],
    'new node test average_precision': [],
    'new node test roc_auc': [],
    'new node test mrr': [],
    'Run cost(/epoch)': [],
    'train time': [],
    'val time': [],
    'load feature time': [],
    'encodeCo time': [],
    'construct patchs time': [],
    'transform time': [],
    'neighbor sample time': [],
    'try time': []
}
df = pd.DataFrame(data)

# 指定文件夹路径
folder_path = 'logs'

# 定义正则表达式模式
patterns = {
    'validate average_precision': r'validate average_precision, (\d+\.\d+)',
    'validate roc_auc': r'validate roc_auc, (\d+\.\d+)',
    'validate mrr': r'validate mrr, (\d+\.\d+)',
    'new node validate average_precision': r'new node validate average_precision, (\d+\.\d+)',
    'new node validate roc_auc': r'new node validate roc_auc, (\d+\.\d+)',
    'new node validate mrr': r'new node validate mrr, (\d+\.\d+)',
    'test average_precision': r'test average_precision, (\d+\.\d+)',
    'test roc_auc': r'test roc_auc, (\d+\.\d+)',
    'test mrr': r'test mrr, (\d+\.\d+)',
    'new node test average_precision': r'new node test average_precision, (\d+\.\d+)',
    'new node test roc_auc': r'new node test roc_auc, (\d+\.\d+)',    
    'new node test mrr': r'new node test mrr, (\d+\.\d+)',
    'Run cost(/epoch)': r'Run \d+ cost (\d+\.\d+) seconds'
}

# 遍历文件夹内的文件
for root, dirs, files in os.walk(folder_path):
    # files = [os.path.join(root, file) for file in files]
    # files.sort(key=os.path.getmtime)
    for file_name in files:
    # if files:
        # file_path = files[-1]
        file_path = os.path.join(root, file_name)
        
        # 读取文件内容
        if(file_path.find('.adapt')!=-1 or file_path.find('.onehop')!=-1 or file_path.find('.norole')!=-1):
            with open(file_path, 'r') as file:
                content = file.read()                    
                matches = re.findall(r'Epoch: (\d+),', content)
                num_epochs = int(matches[-1]) if matches else None
                if num_epochs:
                    print(num_epochs)
                
                # 只取 "get final performance on dataset" 之后的文本
                acc_content = content.split('get final performance on dataset')[-1]
                            
                # 提取匹配的值            
                values = {}
                for key, pattern in patterns.items():
                    match = re.search(pattern, acc_content)
                    if match:
                        values[key] = float(match.group(1))
                        if key == 'Run cost(/epoch)' and  num_epochs!=None:
                            values[key] = values[key] / num_epochs
                
                # 将提取的值添加到字典中
                cols = file_path.split('/')
                dict = {'dataset':cols[2], 'model_seed':cols[3], 'file_name':cols[-1], 'file_path': file_path, **values}
                
                # 提取时间
                # 初始化时间总和和计数器
                times_last = {
                    "train time": 0,
                    "val time": 0,
                    "load feature time": 0,
                    "encodeCo time": 0,
                    "construct patchs time": 0,
                    "transform time": 0,
                    "neighbor sample time": 0,
                    "try time": 0
                }

                # 使用正则表达式提取时间值
                matches = re.finditer(r"(train time|val time|load feature time|encodeCo time|construct patchs time|transform time|neighbor sample time|try time):(\d+.\d+)", content)
                for match in matches:
                    if match:
                        time_type = match.group(1)
                        time_value = float(match.group(2))
                        dict[time_type] = time_value
                    
                # 将这些值添加到DataFrame中
                dict['epoch'] = num_epochs
                df = df._append(dict, ignore_index=True)
                df = df[~df['file_path'].str.contains('old')]
                df = df[~df['file_path'].str.contains('bak')]
                df = df[df['file_path'].str.contains('FFNFormer|QSFormer', na=False)]
                df = df[df['file_path'].str.contains('adapt|onehop|norole', na=False)]
                # df = df.sort_values(by='filepath')

# 将数据帧保存到 Excel 表格
df.to_excel('results_a.xlsx', index=False)
print('Done')

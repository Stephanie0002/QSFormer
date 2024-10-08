#  Copyright (c) Meta Platforms, Inc. and affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import argparse
import itertools
import os
import pandas as pd
import numpy as np
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, help='dataset name')
parser.add_argument('--add_reverse', default=False, action='store_true')

args = parser.parse_args()

print(args)
df = pd.read_csv('./DG_data/{}/{}.csv'.format(args.data, args.data))
num_nodes = max(int(df['source'].max()), int(df['destination'].max())) + 1
print('num_nodes: ', num_nodes)

ext_full_indptr = np.zeros(num_nodes + 1, dtype=np.int32)
ext_full_indices = [[] for _ in range(num_nodes)]
ext_full_ts = [[] for _ in range(num_nodes)]
ext_full_eid = [[] for _ in range(num_nodes)]

for idx, row in tqdm(df.iterrows(), total=len(df)):
    src = int(row['source'])
    dst = int(row['destination'])
    
    ext_full_indices[src].append(dst)
    ext_full_ts[src].append(row['timestamp'])
    ext_full_eid[src].append(idx)
    
    if args.add_reverse:
        ext_full_indices[dst].append(src)
        ext_full_ts[dst].append(row['timestamp'])
        ext_full_eid[dst].append(idx)

for i in tqdm(range(num_nodes)):
    ext_full_indptr[i + 1] = ext_full_indptr[i] + len(ext_full_indices[i])

ext_full_indices = np.array(list(itertools.chain(*ext_full_indices)))
ext_full_ts = np.array(list(itertools.chain(*ext_full_ts)))
ext_full_eid = np.array(list(itertools.chain(*ext_full_eid)))

print('Sorting...')

def tsort(i, indptr, indices, t, eid):
    beg = indptr[i]
    end = indptr[i + 1]
    sidx = np.argsort(t[beg:end])
    indices[beg:end] = indices[beg:end][sidx]
    t[beg:end] = t[beg:end][sidx]
    eid[beg:end] = eid[beg:end][sidx]


for i in tqdm(range(ext_full_indptr.shape[0] - 1)):
    tsort(i, ext_full_indptr, ext_full_indices, ext_full_ts, ext_full_eid)

print('saving...')
out_dir = './processed_data/{}'.format(args.data)
if not os.path.exists(out_dir):
    os.makedirs(out_dir)
np.savez('./processed_data/{}/ext_full.npz'.format(args.data), indptr=ext_full_indptr,
         indices=ext_full_indices, ts=ext_full_ts, eid=ext_full_eid)
import os
import torch
import pickle
from tqdm import tqdm

SRC_DIR = '../preprocessed'

def summerise(root_dir, files, cache_path):
    all_results = []
    for f in tqdm(files):
        with open(os.path.join(root_dir, f), 'rb') as fid:
            data = pickle.load(fid)
        i, j = torch.nonzero(data['prior']).unbind(1)
        labels = data['labels'][i, j]
        unique_classes = torch.unique(j)
        info = dict()
        for c in unique_classes:
            idx = torch.nonzero(j == c).squeeze(1)
            n_p = labels[idx].sum()
            n_n = len(idx) - n_p
            info[str(c)] = (n_p, n_n)
        info['index'] = data['index']
        all_results.append(info)
    with open(cache_path, 'wb') as f:
        pickle.dump(all_results, f, pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    train_dir = os.path.join(SRC_DIR, 'train2015')
    test_dir = os.path.join(SRC_DIR, 'test2015')

    train_files = os.listdir(train_dir)
    test_files = os.listdir(test_dir)

    summerise(train_dir, train_files, './stats_train2015.pkl')
    summerise(test_dir, test_files, './stats_test2015.pkl')
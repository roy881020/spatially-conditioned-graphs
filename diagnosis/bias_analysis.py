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
            info[c.item()] = (n_p.item(), n_n.item())
        info['index'] = data['index']
        all_results.append(info)
    with open(cache_path, 'wb') as f:
        pickle.dump(all_results, f, pickle.HIGHEST_PROTOCOL)

def compute_net_bias(results):
    bias = torch.zeros(117, 2)
    normalised_bias = torch.zeros(117, 2)
    for r in results:
        class_idx = torch.tensor(list(r.keys()))
        stats = torch.tensor(list(r.values()))
        bias[class_idx] += stats
        normalised_bias[class_idx] += (stats / stats.sum())
    return bias, normalised_bias

if __name__ == "__main__":
    train_dir = os.path.join(SRC_DIR, 'train2015')
    test_dir = os.path.join(SRC_DIR, 'test2015')

    train_files = os.listdir(train_dir)
    test_files = os.listdir(test_dir)

    train_cache = './stats_train2015.pkl'
    test_cache = './stats_test2015.pkl'

    if not os.path.exists(train_cache):
        summerise(train_dir, train_files, train_cache)
    if not os.path.exists(test_cache):
        summerise(test_dir, test_files, test_cache)

    # Load total training bias
    with open(train_cache, 'rb') as f:
        bias_train2015 = pickle.load(f)
    # Run analysis
    bias, normalised_bias = compute_net_bias(bias_train2015)

    torch.save(dict(bias=bias, nbias=normalised_bias), 'net_bias.pt')
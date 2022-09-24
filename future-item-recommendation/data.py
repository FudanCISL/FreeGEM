import argparse

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='video', choices=['video', 'game', 'ml-100k', 'ml-1m'])
    parser.add_argument('--attr', action='store_true', help='whether to use attribute information')
    parser.add_argument('--beta', default=10.0, type=float, help='hyperparameter that controls time decay')
    parser.add_argument('--dim0', default=4, type=int, help='k_1')
    parser.add_argument('--dim1', default=1, type=int, help='k_2, k_3, k_4, k_5')
    parser.add_argument('--alpha', default=3, type=int, help='search interval of alpha_1, alpha_2, alpha_3 is [0,1,...,alpha-1]')
    args = parser.parse_args()
    args.split = [0.8, 0.1, 0.1]
    args.topk = 10
    args.dtype = np.float64
    return args


def get_data(args):
    data_file = '../data/{}.csv'.format(args.dataset)
    df = pd.read_csv(data_file, names=['u', 'i', 't'], header=None)
    df['t'] -= df['t'].min()
    user_nums = df['u'].nunique()
    item_nums = df['i'].nunique()
    interaction_nums = len(df)
    train_end_index = int(interaction_nums * args.split[0])
    valid_end_index = int(interaction_nums * (args.split[0] + args.split[1]))
    train_df = df.iloc[:train_end_index]
    valid_df = df.iloc[train_end_index:valid_end_index]
    test_df = df.iloc[valid_end_index:]
    # We can only see the maximum time of the validation set, so `valid_df['t'].max()` is returned
    return user_nums, item_nums, train_df, valid_df, test_df, valid_df['t'].max()


def get_attr(args):
    # Only `ml-100k` and `ml-1m` have attribute information
    if args.dataset == 'ml-100k' or args.dataset == 'ml-1m':
        # Fu
        user_attr_file = '../data/{}_user_attribute.npy'.format(args.dataset)
        Fu = np.load(user_attr_file)
        row, col = Fu.shape
        row_list = []
        col_list = []
        for r in range(row):
            for c in range(col):
                if Fu[r][c] == 1:
                    row_list.append(r)
                    col_list.append(c)
        Fu = coo_matrix((np.ones(len(row_list)), [row_list, col_list]), shape=(row, col))
        # Fi
        item_attr_file = '../data/{}_item_attribute.npy'.format(args.dataset)
        Fi = np.load(item_attr_file)
        row, col = Fi.shape
        row_list = []
        col_list = []
        for r in range(row):
            for c in range(col):
                if Fi[r][c] == 1:
                    row_list.append(r)
                    col_list.append(c)
        Fi = coo_matrix((np.ones(len(row_list), dtype=args.dtype), [row_list, col_list]), shape=(row, col))
        return Fu, Fi
    else:
        return None, None


def build_data(user_nums, item_nums, df0, df1, max_time, args):
    user_list = df0['u'].values
    item_list = df0['i'].values
    time_list = df0['t'].values

    ground_truth = dict()
    for u, i in zip(df1['u'], df1['i']):
        if ground_truth.get(u) is None:
            ground_truth[u] = [i]
        else:
            ground_truth[u].append(i)
    # time decay
    data_train = np.exp(-args.beta * (1 - time_list / max_time), dtype=args.dtype)
    R = coo_matrix((data_train, [user_list, item_list]), shape=(user_nums, item_nums))
    return R, ground_truth

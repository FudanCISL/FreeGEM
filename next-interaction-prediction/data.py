import argparse

import torch
import pandas as pd
import numpy as np
from scipy.sparse import coo_matrix


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['wikipedia', 'lastfm'])
    parser.add_argument('--cuda', default=0, type=int)
    parser.add_argument('--beta', default=0.0, type=float)
    parser.add_argument('--dim', default=512, type=int)
    parser.add_argument('--offline', default=100, type=float)
    parser.add_argument('--lbd', default=0.8, type=float)
    parser.add_argument('--p', default=2, type=int)
    parser.add_argument('--g', default=3, type=int)
    parser.add_argument('--alpha', default=6.0, type=float)
    args = parser.parse_args()
    args.cuda = torch.device('cuda:{}'.format(args.cuda))
    args.split = [0.8, 0.1, 0.1]
    args.topk = 10
    args.dtype = np.float32
    return args


def get_data(args):
    data_file = '../data/{}.csv'.format(args.dataset)
    df = pd.read_csv(data_file, index_col=False, usecols=[0, 1, 2])
    user_nums = df['user_id'].nunique()
    item_nums = df['item_id'].nunique()
    interaction_nums = len(df)
    train_end_index = int(interaction_nums * args.split[0])
    valid_end_index = int(interaction_nums * (args.split[0] + args.split[1]))
    test_end_index = int(interaction_nums * (args.split[0] + args.split[1] + args.split[2]))
    train_df = df.iloc[:train_end_index]
    valid_df = df.iloc[train_end_index:valid_end_index]
    test_df = df.iloc[valid_end_index:test_end_index]
    return user_nums, item_nums, train_df, valid_df, test_df


def build_data(user_list, item_list, time_list, user_nums, item_nums, beta, max_time, args):
    data = np.exp(-beta * (1 - np.array(time_list) / max_time), dtype=args.dtype)
    R = coo_matrix((data, [user_list, item_list]), shape=(user_nums, item_nums))
    return R

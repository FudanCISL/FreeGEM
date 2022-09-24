import time

import torch
import numpy as np
import pandas as pd
from scipy.sparse.linalg import svds
from torch.linalg import svd
from tqdm import tqdm

from attention import Attention
from data import build_data


def iSVD(U, S, V, r, c, v, args):
    """
    This is the Python implementation for incremental SVD paper:
    M. Brand. Fast low-rank modifications of the thin singular value decomposition.
    Linear algebra and its applications, 415(1):20–30, 2006.

    Parameters:
    U - left matrix at t step
    S - degree matrix at t step
    V - right matrix at t step
    r - the row where the update is occur
    c - the colum where the update is occur
    v - the value where the update is occur

    Returns:
	left, degree and right matrix at t+1 step
    """
    m = U.T[:, [r]]
    p = -U @ m
    p[r][0] += np.sqrt(v)
    Ra = (p.T @ p).pow(0.5)
    P = (1 / Ra[0][0]) * p
    n = V.T[:, [c]]
    q = -V @ n
    q[c][0] += np.sqrt(v)
    Rb = (q.T @ q).pow(0.5)
    Q = (1 / Rb[0][0]) * q
    K = torch.cat((torch.cat((torch.diag(S), torch.zeros(m.shape[0], 1).to(args.cuda)), 1), torch.zeros(1, m.shape[0] + 1).to(args.cuda)), 0)
    K += torch.cat((m, torch.tensor([[Ra]]).to(args.cuda)), 0) @ torch.cat((n, torch.tensor([[Rb]]).to(args.cuda)), 0).T
    tUp, tSp, tVhp = svd(K)
    tUp, tSp, tVhp = tUp[:, :-1], tSp[:-1], tVhp[:-1, :]
    Up = torch.cat((U, P), 1) @ tUp
    Sp = tSp
    Vp = torch.cat((V, Q), 1) @ tVhp.T
    return Up, Sp, Vp.T


def new_stage(user_list, item_list, time_list, user_nums, item_nums, T0, args):
    """
    user_list/item_list/time_list is the interaction list visible in the current stage
    """
    max_time = time_list[-1]
    cur_beta = max_time / T0 * args.beta
    R = build_data(user_list, item_list, time_list, user_nums, item_nums, cur_beta, max_time, args)
    u, s, vh = svds(R, k=args.dim, which='LM')
    u, s, vh = torch.from_numpy(u).to(args.cuda), torch.from_numpy(s).to(args.cuda), torch.from_numpy(vh).to(args.cuda)
    start = u * s @ vh
    return max_time, cur_beta, u, s, vh, start


def run(user_nums, item_nums, train_df, valid_df, test_df, args):
    model = Attention().to(args.cuda)
    pointer_P = torch.zeros((user_nums,), dtype=torch.int64)  # indicate how many items each user has interacted with
    pointer_G = 0  # indicate the number of interactions that have occurred in the entire system
    S_P = torch.zeros((user_nums, args.p), dtype=torch.int64)  # store the embedding of item sequence of each user
    S_P_t = torch.from_numpy(np.zeros((user_nums, args.p), dtype=args.dtype)).to(args.cuda)  # record the timestamp of the item sequence that each user interacts with
    S_G = torch.zeros((args.g, user_nums, args.dim))  # store the latest g user embedding
    wG, sG = [], 0.0
    for i in range(1, args.g + 1):
        sG += 1 / i
        wG.append(sG)
    T0 = train_df['timestamp'].values[-1]
    Recall_valid, MRR_valid, Recall_test, MRR_test = 0, 0, 0, 0
    start_time = time.time()
    # here, the initialization is completed and the FreeGEM begin to run
    for phase in ['train', 'test']:
        # in train phase, we can only see the training data
        if phase == 'train':
            df = train_df
            z = zip(valid_df['user_id'], valid_df['item_id'], valid_df['timestamp'])
        # in test phase, we can see the training data and validation data
        else:
            df = pd.concat([train_df, valid_df], axis=0)
            z = zip(test_df['user_id'], test_df['item_id'], test_df['timestamp'])
        user_list, item_list, time_list = list(df['user_id']), list(df['item_id']), list(df['timestamp'])
        # Description of variables in the next line
        # max_time: maximum time of the current stage
        # cur_beta: dynamic time decay coefficient
        # u, s, v: initial SVD results at the current stage
        # start: initial reconstruction results at the current stage
        max_time, cur_beta, u, s, vh, start = new_stage(user_list, item_list, time_list, user_nums, item_nums, T0, args)
        # Frequency-aware Preference Matrix Reconstruction
        EmbU = u * s.pow(1 / args.alpha)
        EmbI = s.pow(1 / args.alpha) * vh.T
        # first record
        S_G[pointer_G % args.g] = EmbU
        pointer_G += 1
        Recall, MRR = 0, 0
        last_Recall, last_MRR = Recall, MRR
        for cnt, tup in tqdm(enumerate(z)):
            uid, iid, t = tup[0], tup[1], tup[2]
            # user long-term preferences vector
            emb_G = torch.zeros((1, args.dim))
            if pointer_G < args.g:
                for _ in range(pointer_G):
                    emb_G += S_G[_][[uid]] * (1 / (pointer_G - _))
            else:
                for _ in range(args.g):
                    emb_G += S_G[(pointer_G + _) % args.g][[uid]] * (1 / (args.g - _))
            emb_G = (emb_G / wG[min(pointer_G, args.g) - 1]).to(args.cuda)
            # user's item sequence
            if pointer_P[uid] < args.p:
                a = torch.exp(-cur_beta * (1 - S_P_t[uid] / max_time))
                b = EmbI[S_P[uid]].T
            else:
                # We use a circular array to store the item sequence, so we need to adjust the position first
                _ = torch.cat((S_P_t[uid][pointer_P[uid] % args.p:], S_P_t[uid][:pointer_P[uid] % args.p]), dim=0)
                a = torch.exp(-cur_beta * (1 - _ / max_time))
                _ = torch.cat((S_P[uid][pointer_P[uid] % args.p:], S_P[uid][:pointer_P[uid] % args.p]), dim=0)
                b = EmbI[_].T
            # user short–term interests vector
            emb_P = model((a * b).T, EmbU[uid])
            # normalize two vectors
            if pointer_P[uid] != 0:  # Maybe the current user appears for the first time
                emb_P = emb_P / torch.sqrt(emb_P.pow(2).sum())
            emb_G = emb_G / torch.sqrt(emb_G.pow(2).sum())
            # user final vector
            emb = args.lbd * emb_P + (1 - args.lbd) * emb_G
            # the scores for predicting the link between the current user and all items
            P = (emb @ EmbI.T).squeeze(0)
            # evaluate
            _, pred_index = P.sort(descending=True)
            if iid in pred_index[: args.topk]:
                Recall += 1
            MRR += 1 / (list(pred_index).index(iid) + 1)
            # update
            user_list.append(uid), item_list.append(iid), time_list.append(t)
            u, s, vh = iSVD(u, s, vh.T, uid, iid, np.exp(-cur_beta * (1 - t / max_time)), args)
            cur = u * s @ vh
            # Monitor: when error exceeds threshold, start the next stage
            if (start - cur).pow(2).sum().pow(1 / 2) > args.offline:
                print('[{}-{:.1f}] {}, {:.3f}'.format(cnt + 1, time.time() - start_time, Recall - last_Recall, MRR - last_MRR))
                last_Recall, last_MRR = Recall, MRR
                max_time, cur_beta, u, s, vh, start = new_stage(user_list, item_list, time_list, user_nums, item_nums, T0, args)
            # update & record
            EmbU = u * s.pow(1 / args.alpha)
            EmbI = s.pow(1 / args.alpha) * vh.T
            S_G[pointer_G % args.g] = EmbU
            pointer_G += 1
            S_P[uid][pointer_P[uid] % args.p] = iid
            S_P_t[uid][pointer_P[uid] % args.p] = t
            pointer_P[uid] += 1
        if phase == 'train':
            Recall_valid = Recall / len(valid_df)
            MRR_valid = MRR / len(valid_df)
            print(Recall_valid, MRR_valid)
        else:
            Recall_test = Recall / len(test_df)
            MRR_test = MRR / len(test_df)
            print(Recall_test, MRR_test)
    return Recall_valid, MRR_valid, Recall_test, MRR_test


def output(Recall_valid, MRR_valid, Recall_test, MRR_test, args):
    Str = '{:.3f}\t'.format(Recall_valid) + \
          '{:.3f}\t'.format(MRR_valid) + \
          '{:.3f}\t'.format(Recall_test) + \
          '{:.3f}\n'.format(MRR_test)
    print(Str)

import torch
import numpy as np
from scipy.sparse.linalg import svds
from scipy.sparse import diags


def call_recall(P, R, ground_truth, args):
    P = torch.from_numpy(P)
    for i, j in zip(R.row, R.col):
        P[i][j] = 1e-18
    _, pred_indexs = P.topk(args.topk)

    Recall = 0.
    for uid in ground_truth:
        pred = pred_indexs[uid].tolist()
        up = len(set(pred) & set(ground_truth[uid]))
        down = len(ground_truth[uid])
        Recall += (up / down)

    Recall /= len(ground_truth)
    return Recall


def run(R, ground_truth, Fu, Fi, alpha_list, args):
    # normalized interaction matrix
    with np.errstate(divide='ignore'):
        Du_ = np.power(R.sum(1).A.T, -1 / 2)[0]
        Di_ = np.power(R.sum(0).A, -1 / 2)[0]
        Di = np.power(R.sum(0).A, 1 / 2)[0]
    R_post = diags(Du_) @ R @ diags(Di_)

    # Use attribute information
    if Fu is not None and Fi is not None and args.attr:
        Au = Fu.T * R_post
        Ai = R_post * Fi

        u0, s0, vh0 = svds(R_post, k=args.dim0, which='LM')  # u  i
        u1, s1, vh1 = svds(Fu, k=args.dim1, which='LM')  # u  fu
        u2, s2, vh2 = svds(Fi, k=args.dim1, which='LM')  # i  fi
        u4, s4, vh4 = svds(Au, k=args.dim1, which='LM')  # fu i
        u5, s5, vh5 = svds(Ai, k=args.dim1, which='LM')  # u  fi

        E0 = u0 * s0 @ vh0 * Di
        E1 = u1 * s1 @ vh1
        E2 = u2 * s2 @ vh2
        E4 = u4 * s4 @ vh4
        E5 = u5 * s5 @ vh5

        R0 = E0  # user --- item
        R1 = E1 @ E4  # user --- user attribute --- item
        R2 = E5 @ E2.T  # user --- item attribute --- item

        if alpha_list == [0, 0, 0]:  # Searching the hyperparameter on the verification set
            best_Recall = 0
            for alpha0 in range(args.alpha):
                for alpha1 in range(args.alpha):
                    for alpha2 in range(args.alpha):
                        P = alpha0 * R0 + alpha1 * R1 + alpha2 * R2
                        Recall = call_recall(P, R, ground_truth, args)
                        if Recall > best_Recall:
                            best_Recall = Recall
                            alpha_list = [alpha0, alpha1, alpha2]
        else:  # Make predictions on the test set using the optimal hyperparameters found on the verification set
            alpha0, alpha1, alpha2 = alpha_list
            P = alpha0 * R0 + alpha1 * R1 + alpha2 * R2
            best_Recall = call_recall(P, R, ground_truth, args)
        return best_Recall, alpha_list
    # Do not use attribute information
    else:
        u, s, vh = svds(R_post, k=args.dim0, which='LM')
        P = u * s @ vh * Di  # inverse normalization
        Recall = call_recall(P, R, ground_truth, args)
        return Recall, alpha_list


def output(Recall_valid, Recall_test):
    Str = '{:.3f}\t'.format(Recall_valid) + \
          '{:.3f}\n'.format(Recall_test)
    print(Str)

import pandas as pd

from data import get_args, get_data, get_attr, build_data
from process import run, output


def call(args):
    user_nums, item_nums, train_df, valid_df, test_df, max_time = get_data(args)
    Fu, Fi = get_attr(args)
    # Used to evaluate the result of the model on the validation set
    R_train, ground_truth_valid = build_data(user_nums, item_nums, train_df, valid_df, max_time, args)
    # Used to evaluate the result of the model on the test set
    R_train_valid, ground_truth_test = build_data(user_nums, item_nums, pd.concat([train_df, valid_df], axis=0), test_df, max_time, args)
    # run FreeGEM on the validation set
    Recall_valid, alpha_list = run(R_train, ground_truth_valid, Fu, Fi, [0, 0, 0], args)
    # run FreeGEM on the test set
    Recall_test, _ = run(R_train_valid, ground_truth_test, Fu, Fi, alpha_list, args)
    output(Recall_valid, Recall_test)


if __name__ == "__main__":
    args = get_args()
    print(args)
    call(args)

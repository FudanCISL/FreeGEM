from data import get_args, get_data
from process import run, output


def call(args):
    user_nums, item_nums, train_df, valid_df, test_df = get_data(args)
    Recall_valid, MRR_valid, Recall_test, MRR_test = run(user_nums, item_nums, train_df, valid_df, test_df, args)
    output(Recall_valid, MRR_valid, Recall_test, MRR_test, args)


if __name__ == "__main__":
    args = get_args()
    print(args)
    call(args)

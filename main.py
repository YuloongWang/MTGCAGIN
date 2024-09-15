import util
from experiment import train
# from analysis import analyze
import os
import csv
import argparse
import warnings
warnings.filterwarnings("ignore")


def parse():
    parser = argparse.ArgumentParser(description='SPATIO-TEMPORAL-ATTENTION-GRAPH-ISOMORPHISM-NETWORK')
    parser.add_argument('-s', '--seed', type=int, default=0)
    parser.add_argument('-n', '--exp_name', type=str, default='stagin_experiment')
    parser.add_argument('-k', '--k_fold', type=int, default=5)
    parser.add_argument('-b', '--minibatch_size', type=int, default=8)
    # D:/ProgramData/BrainGNN/stagin-main/result
    parser.add_argument('-ds', '--sourcedir', type=str, default='./data')
    parser.add_argument('-dt', '--targetdir', type=str, default='./result')

    parser.add_argument('--dataset', type=str, default='rest', choices=['rest', 'task'])
    parser.add_argument('--roi', type=str, default=['aal'], choices=['aal', 'cc200', 'sch400'])
    parser.add_argument('--fwhm', type=float, default=None)

    parser.add_argument('--window_size', type=int, default=20)
    parser.add_argument('--window_stride', type=int, default=1)
    parser.add_argument('--dynamic_length', type=int, default=46)

    parser.add_argument('--lr', type=float, default=0.0005)
    parser.add_argument('--max_lr', type=float, default=0.001)
    parser.add_argument('--reg_lambda', type=float, default=0.00001)
    parser.add_argument('--clip_grad', type=float, default=0.0)
    parser.add_argument('--num_epochs', type=int, default=30)
    parser.add_argument('--num_heads', type=int, default=1)
    parser.add_argument('--num_layers', type=int, default=4, help='K')
    parser.add_argument('--hidden_dim', type=int, default=128, help='D')
    parser.add_argument('--sparsity', type=int, default=30)  # 保留前百分之多少
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--readout', type=str, default='sero', choices=['garo', 'sero', 'mean'])
    parser.add_argument('--cls_token', type=str, default='mean', choices=['sum', 'mean', 'param'])

    parser.add_argument('--num_clusters', type=int, default=7)
    parser.add_argument('--subsample', type=int, default=50)

    parser.add_argument('--no_train', action='store_true')
    parser.add_argument('--no_test', action='store_true')
    parser.add_argument('--no_analysis', action='store_true')

    argv = parser.parse_args()
    argv.targetdir = os.path.join(argv.targetdir, argv.exp_name)
    os.makedirs(argv.targetdir, exist_ok=True)
    # with open(os.path.join(argv.targetdir, 'argv.csv'), 'w', newline='') as f:
    #     writer = csv.writer(f)
    #     writer.writerows(vars(argv).items())
    return argv


if __name__ == '__main__':
    # eFC，gLASSO，BrainGNN
    # parse options and make directories
    # argv = util.option.parse()
    argv = parse()
    # run and analyze experiment
    if not argv.no_train: train(argv)
    # print('test')
    # if not argv.no_test: test(argv)
    # if not argv.no_analysis and argv.roi=='schaefer': analyze(argv)
    exit(0)

# tensorboard --logdir=D:\ProgramData\BrainGNN\stagin-main\result\stagin_experiment\summary
# tensorboard --logdir=C:\Users\wyl79\Desktop\result\stagin_experiment\summary

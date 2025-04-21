import argparse
import logging
import os
import os.path as osp
from lib.configs.config import config, update_config

def parse_args(description=''):
    parser = argparse.ArgumentParser(description=description)
    # general
    parser.add_argument('--cfg', default=None, help='experiment configure file name', type=str)  #
    parser.add_argument('--id', default='', type=str, help='Experiment ID')
    parser.add_argument('--test_dataset', default=None, type=str,  help='Testing Dataset')
    parser.add_argument('--weight_path', help='manually specify model weights', type=str, default='')
    parser.add_argument('--seed', help='random seed', type=int, default=0)

    parser.add_argument('--run', help='run function name', type=str, default='train')
    parser.add_argument('--start_epoch', type = int, default = 1)

    args, rest = parser.parse_known_args()
    update_config(args.cfg, args.id)
    args = parser.parse_args()

    return args, parser

# default complete
def default_complete(config, id):
    project_dir = osp.abspath('.')

    if config.data_dir == '':
        config.data_dir = osp.join(project_dir, 'data')
    if config.model_dir == '':  # checkpoints
        config.model_dir = osp.join(project_dir, 'ckpts/' + id)
    if config.log_dir == '':
        config.log_dir = osp.join(project_dir, 'outputs/' + id)
    return config

args, parser = parse_args()
opt = default_complete(config,  args.id)
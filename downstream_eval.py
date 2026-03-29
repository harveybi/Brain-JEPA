import os
import argparse
import datetime
import yaml
import numpy as np
import json

from downstream_tasks.main_finetune import main as ft_main
from downstream_tasks.main_linprobe import main as lp_main
from src.datasets.downstream_lmdb import FIXED_CHANNELS, FIXED_FRAMES, get_dataset_config_dict

import builtins

builtins.original_print = builtins.print


def get_args_parser():
    parser = argparse.ArgumentParser('Brain-JEPA downstream tasks')
    

    parser.add_argument('--config', default='configs/downstream/fine_tune.yaml', type=str,
                        help='yaml file')
    parser.add_argument('--downstream_task', default='fine_tune', type=str,
                        help='fine_tune or linprobe')
    
    parser.add_argument('--nb_classes', default=2, type=int,
                        help='number of the classification types')
    parser.add_argument('--task', default='classification', type=str,
                        help='number of the classification types')
    
    parser.add_argument('--load_epoch', default='', type=str,
                        help='the epoch to load')
    
    parser.add_argument('--eval', action='store_true', help='number of the classification types')
    

    parser.add_argument('--batch_size', default=20, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=20, type=int)
    parser.add_argument('--blr', default=0.01, type=float)
    parser.add_argument('--min_lr', default=0.000001, type=float)
    parser.add_argument('--smoothing', default=0.0, type=float)
    parser.add_argument('--weight_decay', default=None, type=float)
    parser.add_argument('--layer_decay', default=None, type=float)
    parser.add_argument('--num_seed', default=5, type=int)
    parser.add_argument('--seed', default=0, type=int)
    
    parser.add_argument('--load_path', default='', 
                        help='where to load checkpoint')
    parser.add_argument('--model_name', default='vit_base', type=str, metavar='MODEL', #vit_base_patch16， vit_large_patch16
                        help='Name of model to train')
    parser.add_argument('--data_make_fn', type=str, default='',
                        help='load_dataset')
    
    parser.add_argument('--use_normalization', action='store_true',
                        help='use_normalization')
    parser.add_argument('--downsample', action='store_true',
                        help='downsample')
    parser.add_argument('--add_w', type=str,  default='origin',
                        help='add_w')
    parser.add_argument('--attn_mode', type=str, default='normal',
                        help='attn_mode')
    parser.add_argument('--label_normalization',action='store_true',
                        help='label_normalization')
    parser.add_argument('--gradient_checkpointing',action='store_true',
                        help='gradient_checkpointing')
    
    
    def parse_tuple(arg):
        try:
            return tuple(map(int, arg.split(',')))
        except:
            raise argparse.ArgumentTypeError("Value must be an integer tuple, e.g., 1,2")
        
    parser.add_argument("--crop_size", type=parse_tuple, help="Input a tuple like 1,2")
    parser.add_argument("--patch_size", type=int, default=49, help="Input a tuple like 1,2")
    parser.add_argument("--pred_depth", type=int, default=12, help="Input a tuple like 1,2")
    parser.add_argument("--pred_emb_dim", type=int, default=384, help="Input a tuple like 1,2")
    
    parser.add_argument('--output_root', type=str, default='',
                        help='load_dataset')
    parser.add_argument('--data_root', type=str, default='data',
                        help='root directory containing downstream datasets')
    parser.add_argument('--gradient_csv', type=str, default='',
                        help='path to shared 400-position gradient CSV')
    
    return parser


class Config:
    def __init__(self, dictionary):
        for key, value in dictionary.items():
            setattr(self, key, value)


def load_args_from_yaml(file_path):
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)


def update_config_with_args(config, args):
    for key, value in vars(args).items():
        setattr(config, key, value)
    return config


def format_float(value):
    return f'{value:g}'.replace('.', 'p')


def get_checkpoint_tag(args):
    if os.path.isdir(args.load_path):
        return f'jepa-ep{args.load_epoch}'
    return os.path.basename(args.load_path).split('.')[0]


def apply_dataset_defaults(args):
    dataset_config = get_dataset_config_dict(args.data_make_fn)
    args.dataset_config = dataset_config
    args.task = dataset_config['task']
    args.nb_classes = dataset_config['nb_classes']
    args.crop_size = (FIXED_CHANNELS, FIXED_FRAMES)
    if not args.gradient_csv:
        args.gradient_csv = os.path.join(args.data_root, 'gradient_mapping_400.csv')
    return args


if __name__ == '__main__':
    args_ = get_args_parser()
    args_ = args_.parse_args()
    
    yaml_args = load_args_from_yaml(args_.config)
    config = Config(yaml_args)
    args = update_config_with_args(config, args_)
    args = apply_dataset_defaults(args)
    
    current_time = datetime.datetime.now()
    formatted_time = current_time.strftime('%Y-%m-%d_%H-%M-%S')
    ckpt_tag = get_checkpoint_tag(args)
    run_name = f'{ckpt_tag}_seed{args.seed}_blr{format_float(args.blr)}_{formatted_time}'

    if not os.path.isdir(args.load_path):
        args.finetune = args.load_path
        builtins.print = builtins.original_print
        args.output_dir = os.path.join(args.output_root, args.data_make_fn, args.downstream_task + '_' + args.task, run_name, 'ft_output')
        args.log_dir = os.path.join(args.output_root, args.data_make_fn, args.downstream_task + '_' + args.task, run_name, 'ft_log')
    else:
        ckpt_file = f'jepa-ep{args.load_epoch}.pth.tar'
        builtins.print = builtins.original_print
        args.finetune = os.path.join(args.load_path, ckpt_file)
        args.output_dir = os.path.join(args.output_root, args.data_make_fn, args.downstream_task + '_' + args.task, run_name, 'ft_output')
        args.log_dir = os.path.join(args.output_root, args.data_make_fn, args.downstream_task + '_' + args.task, run_name, 'ft_log')

    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    config_filename = os.path.join(args.output_dir, 'config.yaml')
    with open(config_filename, 'w') as file:
        yaml.safe_dump(vars(args), file, sort_keys=False)

    if args.downstream_task == 'fine_tune':
        ft_main(args)
    else:
        lp_main(args)

                    

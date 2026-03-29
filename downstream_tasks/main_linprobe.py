# --------------------------------------------------------
# References:
# I-JEPA: https://github.com/facebookresearch/ijepa
# MAE: https://github.com/facebookresearch/mae
# --------------------------------------------------------

import datetime
import json
import numpy as np
import os
import sys
import time
import types
from collections import abc

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter

torch_six = types.ModuleType('torch._six')
torch_six.container_abcs = abc
torch_six.string_classes = (str, bytes)
torch_six.int_classes = (int,)
sys.modules.setdefault('torch._six', torch_six)

import timm

assert timm.__version__ == "0.3.2"  # version check
from timm.models.layers import trunc_normal_

import downstream_tasks.util.misc as misc
from downstream_tasks.engine_finetune import evaluate, train_one_epoch
from downstream_tasks.models_vit import VisionTransformer
from downstream_tasks.util.lars import LARS
from downstream_tasks.util.misc import NativeScalerWithGradNormCount as NativeScaler
from src.datasets.downstream_lmdb import make_downstream_dataset


def _filter_checkpoint_by_shape(checkpoint_model, state_dict):
    filtered = {}
    removed = []
    for key, value in checkpoint_model.items():
        new_key = key.replace('module.', 'encoder.')
        if new_key not in state_dict:
            removed.append((new_key, 'missing_in_model'))
            continue
        if state_dict[new_key].shape != value.shape:
            removed.append((new_key, f'{tuple(value.shape)} != {tuple(state_dict[new_key].shape)}'))
            continue
        filtered[new_key] = value
    return filtered, removed


def main(args):
    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    if args.log_dir is not None and not args.eval:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None

    data_loader_train, data_loader_val, data_loader_test, train_dataset, valid_dataset, test_dataset = make_downstream_dataset(
        dataset_name=args.data_make_fn,
        data_root=args.data_root,
        batch_size=args.batch_size,
        pin_mem=args.pin_mem,
        num_workers=args.num_workers,
        drop_last=False,
        use_normalization=args.use_normalization,
    )

    print(f'task: {args.data_make_fn}')
    print(f'len train dataset: {len(train_dataset)}')
    print(f'len validation dataset: {len(valid_dataset)}')
    print(f'len test dataset: {len(test_dataset)}')

    model = VisionTransformer(
        args,
        model_name=args.model_name,
        attn_mode=args.attn_mode,
        num_classes=args.nb_classes,
        global_pool=args.global_pool,
        device=device,
        add_w=args.add_w,
    )

    if args.finetune and not args.eval:
        with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
            f.write(args.finetune + "\n")

        checkpoint = torch.load(args.finetune, map_location='cpu')
        print("Load pre-trained checkpoint from: %s" % args.finetune)

        checkpoint_model = checkpoint['target_encoder']
        state_dict = model.state_dict()
        new_checkpoint_model, removed_keys = _filter_checkpoint_by_shape(checkpoint_model, state_dict)

        for key, reason in removed_keys:
            print(f"Removing key {key} from pretrained checkpoint: {reason}")

        msg = model.load_state_dict(new_checkpoint_model, strict=False)
        print(msg)

        if hasattr(model.head, 'weight'):
            trunc_normal_(model.head.weight, std=0.01)

    model.head = torch.nn.Sequential(
        torch.nn.BatchNorm1d(model.head.in_features, affine=False, eps=1e-6),
        model.head,
    )
    for _, parameter in model.named_parameters():
        parameter.requires_grad = False
    for _, parameter in model.head.named_parameters():
        parameter.requires_grad = True

    model.to(device)

    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("Model = %s" % str(model_without_ddp))
    print('number of params (M): %.6f' % (n_parameters / 1.e6))

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    if args.lr is None:
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)
    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    optimizer = LARS(model_without_ddp.head.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_scaler = NativeScaler()

    if args.task == 'classification':
        criterion = torch.nn.CrossEntropyLoss()
    else:
        criterion = torch.nn.MSELoss()

    print("criterion = %s" % str(criterion))

    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    if args.eval:
        val_stats = evaluate(args, data_loader_val, model, device, args.task)
        print(f"Validation metrics: {val_stats}")
        return

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()

    for epoch in range(args.start_epoch, args.epochs):
        train_stats = train_one_epoch(
            model,
            criterion,
            data_loader_train,
            optimizer,
            device,
            epoch,
            loss_scaler,
            max_norm=None,
            log_writer=log_writer,
            args=args,
        )

        if args.output_dir:
            misc.save_model(
                args=args,
                model=model,
                model_without_ddp=model_without_ddp,
                optimizer=optimizer,
                loss_scaler=loss_scaler,
                epoch=epoch,
            )

        val_stats = evaluate(args, data_loader_val, model, device, args.task)
        test_stats = evaluate(args, data_loader_test, model, device, args.task)

        log_stats = {
            **{f'train_{k}': v for k, v in train_stats.items()},
            **{f'val_{k}': v for k, v in val_stats.items()},
            **{f'test_{k}': v for k, v in test_stats.items()},
            'epoch': epoch,
            'n_parameters': n_parameters,
        }

        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

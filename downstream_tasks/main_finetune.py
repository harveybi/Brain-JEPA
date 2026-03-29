# --------------------------------------------------------
# References:
# I-JEPA: https://github.com/facebookresearch/ijepa
# MAE: https://github.com/facebookresearch/mae
# --------------------------------------------------------

import datetime
import json
import math
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
from timm.loss import LabelSmoothingCrossEntropy
from timm.models.layers import trunc_normal_

import downstream_tasks.util.lr_decay as lrd
import downstream_tasks.util.misc as misc
from downstream_tasks.engine_finetune import evaluate, train_one_epoch
from downstream_tasks.models_vit import VisionTransformer
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


def _is_better_metric(current, best, maximize):
    if current is None:
        return False
    if isinstance(current, float) and math.isnan(current):
        return False
    if best is None:
        return True
    return current > best if maximize else current < best


def _subset_stats(stats, metric_names):
    return {name: float(stats[name]) for name in metric_names if name in stats}


def _save_best_artifacts(args, epoch, model_without_ddp, optimizer, loss_scaler, val_stats, test_stats, n_parameters):
    metric_names = args.dataset_config['logged_metrics']
    summary = {
        'dataset': args.data_make_fn,
        'seed': int(args.seed),
        'blr': float(args.blr),
        'epoch': int(epoch),
        'primary_metric': args.dataset_config['primary_metric'],
        'val': _subset_stats(val_stats, metric_names),
        'test': _subset_stats(test_stats, metric_names),
        'n_parameters': int(n_parameters),
    }

    with open(os.path.join(args.output_dir, 'best_metrics.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)

    checkpoint = {
        'model': model_without_ddp.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
        'scaler': loss_scaler.state_dict() if loss_scaler is not None else None,
        'args': vars(args),
        'best_metrics': summary,
    }
    misc.save_on_master(checkpoint, os.path.join(args.output_dir, 'best_checkpoint.pth'))


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
            trunc_normal_(model.head.weight, std=2e-5)

    model.to(device)

    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("Model = %s" % str(model_without_ddp))
    print('number of params (M): %.2f' % (n_parameters / 1.e6))

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()

    if args.lr is None:
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)
    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    param_groups = lrd.param_groups_lrd(
        model_without_ddp,
        args.weight_decay,
        no_weight_decay_list=model_without_ddp.no_weight_decay(),
        layer_decay=args.layer_decay,
    )

    optimizer = torch.optim.AdamW(param_groups, lr=args.lr)
    loss_scaler = NativeScaler()

    if args.task == 'classification':
        if args.smoothing > 0.:
            criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
        else:
            criterion = torch.nn.CrossEntropyLoss()
    else:
        criterion = torch.nn.MSELoss()

    print("criterion = %s" % str(criterion))

    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    best_val_metric = None

    for epoch in range(args.start_epoch, args.epochs):
        train_stats = train_one_epoch(
            model,
            criterion,
            data_loader_train,
            optimizer,
            device,
            epoch,
            loss_scaler,
            args.clip_grad,
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

        if args.task == 'classification':
            print(f"Validation ACC: {val_stats['acc']:.1f}% | Validation BAC: {val_stats.get('bac', float('nan')):.3f}")
            print(f"Test ACC: {test_stats['acc']:.1f}% | Test BAC: {test_stats.get('bac', float('nan')):.3f}")
        else:
            print(f"Validation MAE: {val_stats['mae']:.3f} | Validation RMSE: {val_stats['rmse']:.3f}")
            print(f"Test MAE: {test_stats['mae']:.3f} | Test RMSE: {test_stats['rmse']:.3f}")

        if log_writer is not None:
            for metric_name in args.dataset_config['logged_metrics']:
                if metric_name in test_stats:
                    log_writer.add_scalar(f'perf/test_{metric_name}', test_stats[metric_name], epoch)
                if metric_name in val_stats:
                    log_writer.add_scalar(f'perf/val_{metric_name}', val_stats[metric_name], epoch)

        log_stats = {
            **{f'train_{k}': v for k, v in train_stats.items()},
            **{f'val_{k}': v for k, v in val_stats.items()},
            **{f'test_{k}': v for k, v in test_stats.items()},
            'epoch': epoch,
            'n_parameters': n_parameters,
        }

        primary_metric = args.dataset_config['primary_metric']
        current_val_metric = float(val_stats[primary_metric])
        is_best = _is_better_metric(
            current=current_val_metric,
            best=best_val_metric,
            maximize=args.dataset_config['maximize_primary_metric'],
        )
        log_stats['is_best'] = is_best

        if is_best:
            best_val_metric = current_val_metric
            if args.output_dir and misc.is_main_process():
                _save_best_artifacts(
                    args=args,
                    epoch=epoch,
                    model_without_ddp=model_without_ddp,
                    optimizer=optimizer,
                    loss_scaler=loss_scaler,
                    val_stats=val_stats,
                    test_stats=test_stats,
                    n_parameters=n_parameters,
                )

        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

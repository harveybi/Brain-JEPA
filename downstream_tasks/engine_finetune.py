# --------------------------------------------------------
# References:
# I-JEPA: https://github.com/facebookresearch/ijepa
# MAE: https://github.com/facebookresearch/mae
# --------------------------------------------------------

import math
import sys
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
)

import downstream_tasks.util.lr_sched as lr_sched
import downstream_tasks.util.misc as misc


def _safe_nan(metric_fn, *args, **kwargs):
    try:
        return float(metric_fn(*args, **kwargs))
    except ValueError:
        return float('nan')


def train_one_epoch(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    loss_scaler,
    max_norm: float = 0,
    log_writer=None,
    args=None,
):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter
    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, (samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
            outputs = model(samples).squeeze()
            if args.task == 'classification' and len(outputs.shape) == 1:
                outputs = outputs.unsqueeze(0)
            if len(targets) == 1:
                loss = criterion(outputs, targets)
            else:
                loss = criterion(outputs, targets.squeeze())
                targets = targets.squeeze()

        loss_value = loss.item()

        if args.task == 'regression':
            mse = torch.mean((targets - outputs.squeeze()) ** 2)
            mae = torch.mean(torch.abs(targets - outputs.squeeze()))

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(
            loss,
            optimizer,
            clip_grad=max_norm,
            parameters=model.parameters(),
            create_graph=False,
            update_grad=(data_iter_step + 1) % accum_iter == 0,
        )
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        if args.task == 'regression':
            metric_logger.update(mse=mse.item())
            metric_logger.update(mae=mae.item())

        max_lr = 0.0
        for group in optimizer.param_groups:
            max_lr = max(max_lr, group["lr"])
        metric_logger.update(lr=max_lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', max_lr, epoch_1000x)

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(args, data_loader, model, device, task):
    criterion = torch.nn.CrossEntropyLoss() if task == 'classification' else torch.nn.MSELoss()
    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Eval:'

    model.eval()

    gt_all = []
    pred_all = []
    prob_all = []

    for images, target in metric_logger.log_every(data_loader, 10, header):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
            output = model(images).squeeze()
            if task == 'classification' and len(output.shape) == 1:
                output = output.unsqueeze(0)
            if len(target) == 1:
                loss = criterion(output, target)
            else:
                loss = criterion(output, target.squeeze())

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())

        if task == 'classification':
            target = target.squeeze()
            probabilities = F.softmax(output, dim=1)
            predictions = torch.argmax(probabilities, dim=1)
            acc = (predictions == target).float().mean().item() * 100.0
            metric_logger.meters['acc'].update(acc, n=batch_size)

            gt_all.append(target.detach().cpu().numpy())
            pred_all.append(predictions.detach().cpu().numpy())
            prob_all.append(probabilities.detach().cpu().numpy())
        else:
            target = target.float().view(-1)
            prediction = output.float().view(-1)
            gt_all.append(target.detach().cpu().numpy())
            pred_all.append(prediction.detach().cpu().numpy())

            mae = torch.mean(torch.abs(target - prediction))
            rmse = torch.sqrt(torch.mean((target - prediction) ** 2))
            metric_logger.update(mae=mae.item(), rmse=rmse.item())

    metric_logger.synchronize_between_processes()

    if task == 'classification':
        gt = np.concatenate(gt_all)
        pred = np.concatenate(pred_all)
        prob = np.concatenate(prob_all, axis=0)
        average = 'binary' if args.nb_classes == 2 else 'macro'

        acc = float((pred == gt).mean() * 100.0)
        f1score = float(f1_score(gt, pred, average=average, zero_division=0))
        bac = float(balanced_accuracy_score(gt, pred))

        results = {
            'loss': metric_logger.loss.global_avg,
            'acc': acc,
            'acc1': acc,
            'f1score': f1score,
            'f1': f1score,
            'bac': bac,
        }

        if args.nb_classes == 2:
            positive_prob = prob[:, 1]
            results['auroc'] = _safe_nan(roc_auc_score, gt, positive_prob)
            results['aucpr'] = _safe_nan(average_precision_score, gt, positive_prob)
        if args.nb_classes > 2 or args.data_make_fn == 'SEEDV':
            results['kappa'] = float(cohen_kappa_score(gt, pred))

        print('Classification metrics:', results)
        return results

    gt = np.concatenate(gt_all)
    pred = np.concatenate(pred_all)
    mae = float(mean_absolute_error(gt, pred))
    rmse = float(mean_squared_error(gt, pred, squared=False))
    mse = float(mean_squared_error(gt, pred))
    results = {
        'loss': metric_logger.loss.global_avg,
        'mse': mse,
        'mae': mae,
        'rmse': rmse,
    }
    print('Regression metrics:', results)
    return results

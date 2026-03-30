import os
import pickle
from dataclasses import asdict, dataclass
from typing import Any, Dict, Tuple

import lmdb
import numpy as np
import torch
import torch.nn.functional as F

# Compatibility shim for records pickled against older NumPy internals.
import numpy.core
import numpy.core.multiarray

import sys

sys.modules.setdefault('numpy._core', numpy.core)
sys.modules.setdefault('numpy._core.multiarray', numpy.core.multiarray)


FIXED_CHANNELS = 400
FIXED_FRAMES = 160


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    task: str
    nb_classes: int
    target_mode: str
    primary_metric: str
    maximize_primary_metric: bool
    logged_metrics: Tuple[str, ...]
    modality: str


DATASET_REGISTRY: Dict[str, DatasetConfig] = {
    'ADNI': DatasetConfig(
        name='ADNI',
        task='classification',
        nb_classes=2,
        target_mode='onehot_argmax',
        primary_metric='bac',
        maximize_primary_metric=True,
        logged_metrics=('acc', 'f1score', 'bac'),
        modality='fMRI',
    ),
    'ADHD200': DatasetConfig(
        name='ADHD200',
        task='classification',
        nb_classes=2,
        target_mode='onehot_argmax',
        primary_metric='bac',
        maximize_primary_metric=True,
        logged_metrics=('acc', 'f1score', 'bac'),
        modality='fMRI',
    ),
    'CamCAN_fMRI_Rest': DatasetConfig(
        name='CamCAN_fMRI_Rest',
        task='regression',
        nb_classes=1,
        target_mode='age_struct',
        primary_metric='mae',
        maximize_primary_metric=False,
        logged_metrics=('mae', 'rmse'),
        modality='fMRI',
    ),
    'CamCAN_MEG_Rest': DatasetConfig(
        name='CamCAN_MEG_Rest',
        task='regression',
        nb_classes=1,
        target_mode='age_struct',
        primary_metric='mae',
        maximize_primary_metric=False,
        logged_metrics=('mae', 'rmse'),
        modality='MEG',
    ),
    'LEMON_fMRI': DatasetConfig(
        name='LEMON_fMRI',
        task='classification',
        nb_classes=2,
        target_mode='lemon_age_group',
        primary_metric='bac',
        maximize_primary_metric=True,
        logged_metrics=('acc', 'f1score', 'bac'),
        modality='fMRI',
    ),
    'LEMON_EEG': DatasetConfig(
        name='LEMON_EEG',
        task='classification',
        nb_classes=2,
        target_mode='lemon_age_group',
        primary_metric='bac',
        maximize_primary_metric=True,
        logged_metrics=('acc', 'f1score', 'bac'),
        modality='EEG',
    ),
    'TUAB': DatasetConfig(
        name='TUAB',
        task='classification',
        nb_classes=2,
        target_mode='onehot_argmax',
        primary_metric='bac',
        maximize_primary_metric=True,
        logged_metrics=('bac', 'aucpr', 'auroc'),
        modality='EEG',
    ),
    'CHB': DatasetConfig(
        name='CHB',
        task='classification',
        nb_classes=2,
        target_mode='onehot_argmax',
        primary_metric='bac',
        maximize_primary_metric=True,
        logged_metrics=('acc', 'f1score', 'bac'),
        modality='EEG',
    ),
    'SEEDV': DatasetConfig(
        name='SEEDV',
        task='classification',
        nb_classes=5,
        target_mode='onehot_argmax',
        primary_metric='kappa',
        maximize_primary_metric=True,
        logged_metrics=('acc', 'f1score', 'bac', 'kappa'),
        modality='EEG',
    ),
}


def get_dataset_config(name: str) -> DatasetConfig:
    if name not in DATASET_REGISTRY:
        valid = ', '.join(sorted(DATASET_REGISTRY))
        raise ValueError(f'Unknown downstream dataset "{name}". Valid values: {valid}')
    return DATASET_REGISTRY[name]


def get_dataset_config_dict(name: str) -> Dict[str, Any]:
    return asdict(get_dataset_config(name))


class BrainSignalLMDBDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        lmdb_path: str,
        dataset_config: DatasetConfig,
        use_normalization: bool = False,
        fixed_channels: int = FIXED_CHANNELS,
        fixed_frames: int = FIXED_FRAMES,
    ):
        self.lmdb_path = lmdb_path
        self.dataset_config = dataset_config
        self.use_normalization = use_normalization
        self.fixed_channels = fixed_channels
        self.fixed_frames = fixed_frames
        self.env = None

        with lmdb.open(self.lmdb_path, readonly=True, lock=False, readahead=False, subdir=os.path.isdir(self.lmdb_path)) as env:
            with env.begin() as txn:
                self.keys = [item['key'].encode('utf-8') for item in pickle.loads(txn.get(b'__keys__'))]

    def __len__(self):
        return len(self.keys)

    def __getstate__(self):
        state = self.__dict__.copy()
        state['env'] = None
        return state

    def _get_env(self):
        if self.env is None:
            self.env = lmdb.open(
                self.lmdb_path,
                readonly=True,
                lock=False,
                readahead=False,
                max_readers=1,
                subdir=os.path.isdir(self.lmdb_path),
            )
        return self.env

    def _prepare_signal(self, signal: np.ndarray) -> torch.Tensor:
        signal = np.asarray(signal, dtype=np.float32)
        rows, frames = signal.shape

        padded = np.zeros((self.fixed_channels, frames), dtype=np.float32)
        padded[: min(rows, self.fixed_channels), :] = signal[: self.fixed_channels, :]
        tensor = torch.from_numpy(padded)

        if self.use_normalization:
            mean = tensor.mean()
            std = tensor.std()
            if torch.isfinite(std) and std.item() > 0:
                tensor = (tensor - mean) / std
            else:
                tensor = tensor - mean

        if frames > self.fixed_frames:
            tensor = F.interpolate(
                tensor.unsqueeze(0),
                size=self.fixed_frames,
                mode='linear',
                align_corners=False,
            ).squeeze(0)
        elif frames < self.fixed_frames:
            padded_time = torch.zeros((self.fixed_channels, self.fixed_frames), dtype=torch.float32)
            padded_time[:, :frames] = tensor
            tensor = padded_time

        return tensor.unsqueeze(0).to(torch.float32)

    def _extract_target(self, raw_target: Any):
        if self.dataset_config.target_mode == 'onehot_argmax':
            return int(np.argmax(np.asarray(raw_target)))
        if self.dataset_config.target_mode == 'lemon_age_group':
            return int(np.argmax(np.asarray(raw_target)[0]))
        if self.dataset_config.target_mode == 'age_struct':
            if hasattr(raw_target, 'dtype') and getattr(raw_target.dtype, 'fields', None):
                age = np.asarray(raw_target['age']).reshape(-1)[0]
            else:
                age = np.asarray(raw_target).reshape(-1)[0]
            return float(age)
        raise ValueError(f'Unsupported target_mode "{self.dataset_config.target_mode}"')

    def __getitem__(self, idx):
        with self._get_env().begin() as txn:
            record = pickle.loads(txn.get(self.keys[idx]))

        signal = self._prepare_signal(record['signal'])
        target = self._extract_target(record['y'])
        return signal, target


def make_downstream_dataset(
    dataset_name: str,
    data_root: str,
    batch_size: int,
    pin_mem: bool = True,
    num_workers: int = 8,
    drop_last: bool = False,
    use_normalization: bool = False,
):
    dataset_config = get_dataset_config(dataset_name)
    base_dir = os.path.join(data_root, dataset_name)

    def build_dataset(split: str):
        return BrainSignalLMDBDataset(
            lmdb_path=os.path.join(base_dir, split, 'BrainSignal.lmdb'),
            dataset_config=dataset_config,
            use_normalization=use_normalization,
        )

    train_dataset = build_dataset('train')  
    valid_dataset = build_dataset('val')
    test_dataset = build_dataset('test')

    # train_dataset = train_dataset + test_dataset

    loader_kwargs = {
        'batch_size': batch_size,
        'drop_last': drop_last,
        'pin_memory': pin_mem,
        'num_workers': num_workers,
        'persistent_workers': num_workers > 0,
    }

    train_data_loader = torch.utils.data.DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    valid_data_loader = torch.utils.data.DataLoader(valid_dataset, shuffle=False, **loader_kwargs)
    test_data_loader = torch.utils.data.DataLoader(test_dataset, shuffle=False, **loader_kwargs)

    return train_data_loader, valid_data_loader, test_data_loader, train_dataset, valid_dataset, test_dataset

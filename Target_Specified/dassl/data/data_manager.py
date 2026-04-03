import torch
import torchvision.transforms as T
from tabulate import tabulate
from torch.utils.data import Dataset as TorchDataset
import numpy as np
from dassl.utils import read_image

from .datasets import build_dataset
from .samplers import build_sampler
from .transforms import INTERPOLATION_MODES, build_transform
from PIL import Image


def build_data_loader(
    cfg,
    sampler_type="SequentialSampler",
    data_source=None,
    batch_size=64,
    n_domain=0,
    n_ins=2,
    tfm=None,
    is_train=True,
    dataset_wrapper=None,
    mode="norm"
):
    sampler = build_sampler(
        sampler_type,
        cfg=cfg,
        data_source=data_source,
        batch_size=batch_size,
        n_domain=n_domain,
        n_ins=n_ins
    )
    dataset_wrapper = DatasetWrapper

    data_loader = torch.utils.data.DataLoader(
        dataset_wrapper(cfg, data_source, transform=tfm, is_train=is_train, mode=mode),
        batch_size=batch_size,
        sampler=sampler,
        num_workers=cfg.DATALOADER.NUM_WORKERS,
        drop_last=is_train and len(data_source) >= batch_size,
        pin_memory=(torch.cuda.is_available() and cfg.USE_CUDA)
    )
    assert len(data_loader) > 0

    return data_loader


class DataManager:

    def __init__(
        self,
        cfg,
        custom_tfm_train=None,
        custom_tfm_test=None,
        dataset_wrapper=None
    ):
        dataset = build_dataset(cfg)

        if custom_tfm_train is None:
            tfm_train = build_transform(cfg, is_train=True)
        else:
            print("* Using custom transform for training")
            tfm_train = custom_tfm_train


        tfm_free = build_transform(cfg, is_expend=True)

        if custom_tfm_test is None:
            tfm_test = build_transform(cfg, is_train=False)
        else:
            print("* Using custom transform for testing")
            tfm_test = custom_tfm_test

        train_loader_x = build_data_loader(
            cfg,
            sampler_type=cfg.DATALOADER.TRAIN_X.SAMPLER,
            data_source=dataset.train_x,
            batch_size=cfg.DATALOADER.TRAIN_X.BATCH_SIZE,
            n_domain=cfg.DATALOADER.TRAIN_X.N_DOMAIN,
            n_ins=cfg.DATALOADER.TRAIN_X.N_INS,
            tfm=tfm_train,
            is_train=True,
            dataset_wrapper=dataset_wrapper
        )

        train_loader_e = build_data_loader(
            cfg,
            sampler_type=cfg.DATALOADER.TRAIN_X.SAMPLER,
            data_source=dataset.train_x,
            batch_size=cfg.DATALOADER.TRAIN_X.BATCH_SIZE,
            n_domain=cfg.DATALOADER.TRAIN_X.N_DOMAIN,
            n_ins=cfg.DATALOADER.TRAIN_X.N_INS,
            tfm=tfm_free,
            is_train=True,
            dataset_wrapper=dataset_wrapper
        )

        train_loader_u = build_data_loader(
            cfg,
            sampler_type=cfg.DATALOADER.TRAIN_U.SAMPLER,
            data_source=dataset.train_u,
            batch_size=cfg.DATALOADER.TRAIN_U.BATCH_SIZE,
            n_domain=cfg.DATALOADER.TRAIN_U.N_DOMAIN,
            n_ins=cfg.DATALOADER.TRAIN_U.N_INS,
            tfm=tfm_train,
            is_train=True,
            dataset_wrapper=dataset_wrapper
        )

        test_loader_x = build_data_loader(
            cfg,
            sampler_type=cfg.DATALOADER.TEST.SAMPLER,
            data_source=dataset.test_x,
            batch_size=cfg.DATALOADER.TEST.BATCH_SIZE,
            tfm=tfm_test,
            is_train=False,
            dataset_wrapper=dataset_wrapper
        )

        test_loader_u = build_data_loader(
            cfg,
            sampler_type=cfg.DATALOADER.TEST.SAMPLER,
            data_source=dataset.test_u,
            batch_size=cfg.DATALOADER.TEST.BATCH_SIZE,
            tfm=tfm_test,
            is_train=False,
            dataset_wrapper=dataset_wrapper
        )

        self._num_classes = dataset.num_classes
        self._num_source_domains = len(cfg.DATASET.SOURCE_DOMAINS)
        self._lab2cname = dataset.lab2cname

        self.dataset = dataset
        self.train_loader_x = train_loader_x
        self.train_loader_u = train_loader_u
        self.train_loader_e = train_loader_e
        self.test_loader_x = test_loader_x
        self.test_loader_u = test_loader_u

        if cfg.VERBOSE:
            self.show_dataset_summary(cfg)

    @property
    def num_classes(self):
        return self._num_classes

    @property
    def num_source_domains(self):
        return self._num_source_domains

    @property
    def lab2cname(self):
        return self._lab2cname

    def show_dataset_summary(self, cfg):
        dataset_name = cfg.DATASET.NAME
        source_domains = cfg.DATASET.SOURCE_DOMAINS
        target_domains = cfg.DATASET.TARGET_DOMAINS

        table = []
        table.append(["Dataset", dataset_name])
        if source_domains:
            table.append(["Source", source_domains])
        if target_domains:
            table.append(["Target", target_domains])
        table.append(["# classes", f"{self.num_classes:,}"])
        table.append(["# train_x", self.dataset.train_x[0].domain, f"{len(self.dataset.train_x):,}"])
        table.append(["# train_e", self.dataset.train_x[0].domain, f"{len(self.dataset.train_x):,}"])
        table.append(["# train_u", self.dataset.train_u[0].domain, f"{len(self.dataset.train_u):,}"])
        table.append(["# test_x", self.dataset.test_x[0].domain, f"{len(self.dataset.test_x):,}"])
        table.append(["# test_u", self.dataset.test_u[0].domain, f"{len(self.dataset.test_u):,}"])
        print(tabulate(table))


class DatasetWrapper(TorchDataset):

    def __init__(self, cfg, data_source, transform=None, is_train=False, mode="norm"):
        self.cfg = cfg
        self.data_source = data_source
        self.transform = transform
        self.is_train = is_train
        self.mode = mode
        self.k_tfm = cfg.DATALOADER.K_TRANSFORMS if is_train else 1

        if self.k_tfm > 1 and transform is None:
            raise ValueError(
                "Cannot augment the image {} times "
                "because transform is None".format(self.k_tfm)
            )

        interp_mode = INTERPOLATION_MODES[cfg.INPUT.INTERPOLATION]
        to_tensor = []
        to_tensor += [T.Resize(cfg.INPUT.SIZE, interpolation=interp_mode)]
        to_tensor += [T.ToTensor()]
        if "normalize" in cfg.INPUT.TRANSFORMS:
            normalize = T.Normalize(
                mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD
            )
            to_tensor += [normalize]
        self.to_tensor = T.Compose(to_tensor)

    def __len__(self):
        return len(self.data_source)

    def __getitem__(self, idx):
        item = self.data_source[idx]

        output = {
            "label": item.label,
            "domain": item.domain,
            "impath": item.impath,
            "index": idx
        }

        img0 = read_image(item.impath)
        img_tr = self._transform_image(self.transform, img0)
        if self.mode == "norm":
            img = img_tr
        elif self.mode == "watermark":
            img = self._add_watermark(img_tr)

        output["img"] = img


        return output

    def _transform_image(self, tfm, img0):
        img_list = []
        for k in range(self.k_tfm):
            img_list.append(tfm(img0))

        img = img_list
        if len(img) == 1:
            img = img[0]

        return img

    def _add_watermark(self, img0):
        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(3, 1, 1)
        std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(3, 1, 1)
        img_denorm = img0 * std + mean
        img_denorm = torch.clamp(img_denorm, 0, 1)
        img_np = (img_denorm.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        mask = np.zeros(img_np.shape[:2], dtype=np.uint8)
        mask[::2, :] = 255
        mask[:, ::2] = 255

        img_mask = np.minimum(img_np.astype(int) + mask[:, :, None], 255).astype(np.uint8)
        img_tensor = torch.from_numpy(img_mask).permute(2, 0, 1).float() / 255.0
        img_tensor = (img_tensor - mean) / std
        return img_tensor


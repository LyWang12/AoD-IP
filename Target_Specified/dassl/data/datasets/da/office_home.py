import os.path as osp
import random
from dassl.utils import listdir_nohidden, set_random_seed

from ..build import DATASET_REGISTRY
from ..base_dataset import Datum, DatasetBase


@DATASET_REGISTRY.register()
class OfficeHome(DatasetBase):
    """Office-Home.

    Statistics:
        - Around 15,500 images.
        - 65 classes related to office and home objects.
        - 4 domains: Art, Clipart, Product, Real World.
        - URL: http://hemanthdv.org/OfficeHome-Dataset/.

    Reference:
        - Venkateswara et al. Deep Hashing Network for Unsupervised
        Domain Adaptation. CVPR 2017.
    """

    dataset_dir = "office_home"
    domains = ["art", "clipart", "product", "real_world"]

    def __init__(self, cfg):
        root = osp.abspath(osp.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.split_dir = osp.join(self.dataset_dir, "split_random")
        
        self.check_input_domains(
            cfg.DATASET.SOURCE_DOMAINS, cfg.DATASET.TARGET_DOMAINS
        )
        train_x = self._read_data(cfg.DATASET.SOURCE_DOMAINS, split="train")
        train_expend = self._read_data(cfg.DATASET.SOURCE_DOMAINS, split="train")
        train_u = self._read_data(cfg.DATASET.TARGET_DOMAINS, split="train")
        test_x = self._read_data(cfg.DATASET.SOURCE_DOMAINS, split="test")
        test_u = self._read_data(cfg.DATASET.TARGET_DOMAINS, split="test")

        super().__init__(train_x=train_x, train_u=train_u, train_expend=train_expend, test_x=test_x, test_u=test_u)

    def _read_data(self, input_domains, split="train"):
        items = []

        for domain, dname in enumerate(input_domains):
            filename = dname + "_" + split + ".txt"
            split_file = osp.join(self.split_dir, filename)

            with open(split_file, "r") as f:
                lines = f.readlines()
                for line in lines:
                    line = line.strip()
                    impath, label = line.split(" ")
                    classname = impath.split("/")[1]
                    impath = osp.join(self.dataset_dir, impath)
                    label = int(label)
                    item = Datum(
                        impath=impath,
                        label=label,
                        domain=dname,
                        classname=classname
                    )
                    items.append(item)
        return items

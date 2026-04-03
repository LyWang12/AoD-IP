import os
import random
import os.path as osp
import tarfile
import zipfile
from collections import defaultdict
import gdown

from dassl.utils import check_isfile


class Datum:

    def __init__(self, impath="", label=0, domain=0, classname=""):
        assert isinstance(impath, str)
        assert check_isfile(impath)

        self._impath = impath
        self._label = label
        self._domain = domain
        self._classname = classname

    @property
    def impath(self):
        return self._impath

    @property
    def label(self):
        return self._label

    @property
    def domain(self):
        return self._domain

    @property
    def classname(self):
        return self._classname


class DatasetBase:
    dataset_dir = ""
    domains = []

    def __init__(self, train_x=None, train_expend=None, test_x = None,
                 add_test_1 = None, add_test_2 = None, add_test_3 = None, add_test_4 = None):
        self._train_x = train_x
        self._train_expend = train_expend
        self._test_x = test_x
        self._add_test_1 = add_test_1
        self._add_test_2 = add_test_2
        self._add_test_3 = add_test_3
        self._add_test_4 = add_test_4
        self._num_classes = self.get_num_classes(train_x)
        self._lab2cname, self._classnames = self.get_lab2cname(train_x)

    @property
    def train_x(self):
        return self._train_x

    @property
    def train_expend(self):
        return self._train_expend

    @property
    def test_x(self):
        return self._test_x

    @property
    def add_test_1(self):
        return self._add_test_1

    @property
    def add_test_2(self):
        return self._add_test_2

    @property
    def add_test_3(self):
        return self._add_test_3

    @property
    def add_test_4(self):
        return self._add_test_4

    @property
    def lab2cname(self):
        return self._lab2cname

    @property
    def classnames(self):
        return self._classnames

    @property
    def num_classes(self):
        return self._num_classes

    @staticmethod
    def get_num_classes(data_source):
        label_set = set()
        for item in data_source:
            label_set.add(item.label)
        return max(label_set) + 1

    @staticmethod
    def get_lab2cname(data_source):
        container = set()
        for item in data_source:
            container.add((item.label, item.classname))
        mapping = {label: classname for label, classname in container}
        labels = list(mapping.keys())
        labels.sort()
        classnames = [mapping[label] for label in labels]
        return mapping, classnames

    def check_input_domains(self, source_domains, target_domains):
        assert len(source_domains) > 0, "source_domains (list) is empty"
        assert len(target_domains) > 0, "target_domains (list) is empty"
        self.is_input_domain_valid(source_domains)
        self.is_input_domain_valid(target_domains)

    def is_input_domain_valid(self, input_domains):
        for domain in input_domains:
            if domain not in self.domains:
                raise ValueError(
                    "Input domain must belong to {}, "
                    "but got [{}]".format(self.domains, domain)
                )
import argparse
import torch
import os

from dassl.utils import setup_logger, collect_env_info
from dassl.config import get_cfg_default
from dassl.engine import build_trainer

from dassl.data.datasets import Office31
from dassl.data.datasets import OfficeHome
from dassl.data.datasets import miniDomainNet

import trainers.ipclip_vitB16

def print_args(args, cfg):
    print("***************")
    print("** Arguments **")
    print("***************")
    optkeys = list(args.__dict__.keys())
    optkeys.sort()
    for key in optkeys:
        print("{}: {}".format(key, args.__dict__[key]))
    print("************")
    print("** Config **")
    print("************")
    print(cfg)


def reset_cfg(cfg, args):
    if args.root:
        cfg.DATASET.ROOT = args.root

    if args.epoch:
        cfg.OPTIM.MAX_EPOCH = args.epoch

    if args.output_dir:
        cfg.OUTPUT_DIR = args.output_dir

    if args.resume:
        cfg.RESUME = args.resume

    if args.load_epoch:
        cfg.RESUME = args.load_epoch

    if args.batch:
        cfg.DATALOADER.TRAIN_X.BATCH_SIZE = args.batch
        cfg.DATALOADER.TRAIN_U.BATCH_SIZE = args.batch

    if args.transforms:
        cfg.INPUT.TRANSFORMS = args.transforms

    if args.trainer:
        cfg.TRAINER.NAME = args.trainer

    if args.backbone:
        cfg.MODEL.BACKBONE.NAME = args.backbone

    if args.head:
        cfg.MODEL.HEAD.NAME = args.head


def extend_cfg(cfg):

    from yacs.config import CfgNode as CN

    cfg.MODEL.BACKBONE.PATH = "./assets"
    cfg.DATASET.TRAIN_EPS = 16
    cfg.DATASET.TEST_EPS = 8

    cfg.TRAINER.IPCLIPB16 = CN()
    cfg.TRAINER.IPCLIPB16.PREC = "amp"



def setup_cfg(args):
    cfg = get_cfg_default()
    extend_cfg(cfg)

    if args.dataset_config_file:
        cfg.merge_from_file(args.dataset_config_file)

    if args.config_file:
        cfg.merge_from_file(args.config_file)

    reset_cfg(cfg, args)

    cfg.merge_from_list(args.opts)

    cfg.freeze()

    return cfg


def main(args):
    cfg = setup_cfg(args)
    setup_logger(cfg.OUTPUT_DIR)
    if torch.cuda.is_available() and cfg.USE_CUDA:
        torch.backends.cudnn.benchmark = True

    trainer = build_trainer(cfg)

    if not args.no_train:
        print("No! Training")
        trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=str, default="./output/officehome/A-C", help="output directory")
    parser.add_argument("--resume", type=str, default="", help="checkpoint directory (from which the training resumes)", )
    parser.add_argument("--dataset-config-file", type=str, default="./configs/datasets/officehomeAC.yaml",  help="path to config file for dataset setup", )
    parser.add_argument("--epoch", type=int, default=20)
    parser.add_argument("--batch", type=int, default=3, help="divisible by the number of GPUs")
    parser.add_argument("--trainer", type=str, default="IPCLIPB16", help="name of trainer")
    parser.add_argument("--root", type=str, default="../Datasets", help="path to dataset")
    parser.add_argument("--transforms", type=str, nargs="+", help="data augmentation methods")
    parser.add_argument("--config-file", type=str, default="configs/trainer/vitB16.yaml", help="path to config file")
    parser.add_argument("--backbone", type=str, default="", help="name of CNN backbone")
    parser.add_argument("--head", type=str, default="", help="name of head")
    parser.add_argument("--load-epoch", type=int, help="load model weights at this epoch for evaluation")
    parser.add_argument("--model-dir", type=str, default="", help="load model from this directory for eval-only mode", )
    parser.add_argument("--no-train", action="store_true", help="do not call trainer.train()")
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER, help="modify config options using the command-line", )
    args = parser.parse_args()
    main(args)


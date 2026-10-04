import os
import argparse
import torch
from dassl.utils import setup_logger, set_random_seed
from dassl.config import get_cfg_default
from dassl.engine import build_trainer
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

    if args.seed:
        cfg.SEED = args.seed

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
    if cfg.SEED >= 0:
        print("Setting fixed seed: {}".format(cfg.SEED))
        set_random_seed(cfg.SEED)
    setup_logger(cfg.OUTPUT_DIR)
    if torch.cuda.is_available() and cfg.USE_CUDA:
        torch.backends.cudnn.benchmark = True

    trainer = build_trainer(cfg)

    if not args.no_train:
        print("No! Training")
        trainer.train()


if __name__ == "__main__":
    DATASETS = {  # name: (dataset config prefix, default train batch = largest that fits one 96 GB GPU)
        "officehome": ("officehome", 24),      # tasks: A-C A-P A-R C-A C-P C-R P-A P-C P-R R-A R-C R-P
        "domainnet": ("mini_domainnet", 12),   # tasks: C-P C-R C-S P-C P-R P-S R-C R-P R-S S-C S-P S-R
        "office31": ("office31", 48),          # tasks: A-D A-W D-A D-W W-A W-D
    }
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="officehome", choices=DATASETS)
    parser.add_argument("--task", type=str, default="A-C", help="source-target pair, e.g. A-C")
    parser.add_argument("--resume", type=str, default="", help="checkpoint directory (from which the training resumes)", )
    parser.add_argument("--epoch", type=int, default=10)
    ####################
    parser.add_argument("--seed", type=int, default=1, help="only positive value enables a fixed seed")
    parser.add_argument("--batch", type=int, default=None, help="train batch size; default: per-dataset value in DATASETS")
    parser.add_argument("--trainer", type=str, default="IPCLIPB16", help="name of trainer")
    parser.add_argument("--root", type=str, default="../Datasets", help="path to dataset")
    parser.add_argument("--transforms", type=str, nargs="+", help="data augmentation methods")
    parser.add_argument("--config-file", type=str, default="./configs/trainer/vitB16.yaml", help="path to config file")
    parser.add_argument("--backbone", type=str, default="", help="name of CNN backbone")
    parser.add_argument("--head", type=str, default="", help="name of head")
    parser.add_argument("--load-epoch", type=int, help="load model weights at this epoch for evaluation")
    parser.add_argument("--model-dir", type=str, default="", help="load model from this directory for eval-only mode", )
    parser.add_argument("--no-train", action="store_true", help="do not call trainer.train()")
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER, help="modify config options using the command-line", )
    args = parser.parse_args()
    cfg_prefix, default_batch = DATASETS[args.dataset]
    args.batch = args.batch or default_batch
    args.output_dir = f"./output/{args.dataset}/{args.task}/seed_{args.seed}"
    args.dataset_config_file = f"./configs/datasets/{cfg_prefix}{args.task.replace('-', '')}.yaml"
    main(args)

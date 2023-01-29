#import sys
#sys.path.append('./')

import logging
import os
import sys
import random
from datetime import datetime
import shutil
from pathlib import Path
import argparse
from time import time

import numpy as np
import torch
from torch import optim
from torch.cuda.amp import GradScaler

from open_clip import create_model_and_transforms, trace_model, get_tokenizer
from open_clip import ClipLoss, get_cast_dtype
from training.data import get_data
from training.distributed import is_master, init_distributed_device, world_info_from_env
from training.logger import setup_logging
from training.params import parse_args
from training.scheduler import cosine_lr
from training.train import train_one_epoch, evaluate

from argparse import Namespace

import smdebug.pytorch as smd
from smdebug.core.reduction_config import ReductionConfig
from smdebug.core.save_config import SaveConfig
from smdebug.core.collection import CollectionKeys
from smdebug.core.config_constants import DEFAULT_CONFIG_FILE_PATH

def parse_args():
    cmdline = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    #cmdline.add_argument('--train-data', default='/opt/ml/data/train/',
    #                     help="""train data location""")
    cmdline.add_argument('--train-num-samples', default=10000000, type=int,
                         help="""number of training samples per epoch""")
    #cmdline.add_argument('--val-data', default='/opt/ml/data/val/',
    #                     help="""val data location""")
    cmdline.add_argument('--val-num-samples', default=10000, type=int,
                         help="""number of val samples per epoch""")
    cmdline.add_argument('--batch-size', default=64, type=int,
                         help="""batch size per device""")
    cmdline.add_argument('--warmup', default=2000, type=int,
                         help="""warmup steps""")
    cmdline.add_argument('--epochs', default=20, type=int,
                         help="""number of epochs""")
    cmdline.add_argument('--lr', default=5e-4, type=float,
                         help="""learning rate""")
    cmdline.add_argument('--wd', default=0.2, type=float,
                         help="""weight decay""")
    cmdline.add_argument('--eps', default=1.0e-6, type=float,
                         help="""optimizer epsilon""")
    cmdline.add_argument('--beta1', default=0.9, type=float,
                         help="""optimizer beta 1""")
    cmdline.add_argument('--beta2', default=0.98, type=float,
                         help="""optimizer beta 2""")
    cmdline.add_argument('--precision', default='amp',
                         help="""precision amp or bf16""")
    cmdline.add_argument('--workers', default=4, type=int,
                         help="""dataloader workers""")
    cmdline.add_argument('--name', default='proto_1',
                         help="""training name""")
    cmdline.add_argument('--log-every-n-steps', default=100, type=int,
                         help="""logging frequency""")
    cmdline.add_argument('--log-path', default="./training_log",
                         help="""logging path""")
    cmdline.add_argument('--dist-backend', default='nccl',
                         help="""distribution type""")
    return cmdline


arg_dict = {
    "train_data": "pipe:aws s3 cp s3://jbsnyder-sagemaker-us-west-2/data/laion/laion400m/data/{00000..02499}.tar -",
    "train_num_samples": 8000000,
    "val_data": "pipe:aws s3 cp s3://jbsnyder-sagemaker-us-west-2/data/laion/laion400m/data/{02500..02587}.tar -",
    "val_num_samples": 10000,
    "imagenet_val": None, # "pipe:aws s3 cp s3://jbsnyder-sagemaker-us-west-2/data/imagenet/val/val_{0000..0127}.tar -",
    "imagenet_v2": None,
    "dataset_type": "webdataset",
    "batch_size": 64,
    "warmup": 2000,
    "epochs": 100,
    "lr": 5e-4,
    "wd": 0.2,
    "eps": 1.0e-8,
    "beta1": 0.9,
    "beta2": 0.98,
    "precision": "amp",
    "workers": 4,
    "model": "ViT-H-14",
    "name": "proto_1",
    "pretrained": "laion2b_s32b_b79k", #'laion2b_s34b_b79k', #"laion400m_e32",
    "horovod": False,
    "torchscript": False,
    "force_quick_gelu": False,
    "force_custom_text": False,
    "force_patch_dropout": None,
    "pretrained_image": False,
    "image_mean": None,
    "image_std": None,
    "use_bn_sync": True,
    "seed": 0,
    "accum_freq": 1,
    "local_loss": False,
    "gather_with_grad": False,
    "skip_scheduler": False,
    "grad_clip_norm": None,
    "log_every_n_steps": 100,
    "wandb": False,
    "log_level": logging.INFO,
    "log_path": "./training_log",
    "debug": False,
    "dist_backend": "nccl",
    "dist_url": "env://",
    "no_set_device_rank": False,
    "ddp_static_graph": False,
    "val_frequency": 1,
}

def main(args):
    
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    
    args.local_rank, args.rank, args.world_size = world_info_from_env()
    args.distributed = args.world_size>1

    args.log_level = logging.DEBUG if args.debug else logging.INFO
    setup_logging(args.log_path, args.log_level)

    device = init_distributed_device(args)

    model, preprocess_train, preprocess_val = create_model_and_transforms(
            args.model,
            args.pretrained,
            precision=args.precision,
            device=device,
            jit=args.torchscript,
            force_quick_gelu=args.force_quick_gelu,
            force_custom_text=args.force_custom_text,
            force_patch_dropout=args.force_patch_dropout,
            pretrained_image=args.pretrained_image,
            image_mean=args.image_mean,
            image_std=args.image_std,
        )

    if args.distributed and not args.horovod:
        if args.use_bn_sync:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        ddp_args = {}
        if args.ddp_static_graph:
            ddp_args['static_graph'] = True
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device], **ddp_args)

    exclude = lambda n, p: p.ndim < 2 or "bn" in n or "ln" in n or "bias" in n or 'logit_scale' in n
    include = lambda n, p: not exclude(n, p)

    named_parameters = list(model.named_parameters())
    gain_or_bias_params = [p for n, p in named_parameters if exclude(n, p) and p.requires_grad]
    rest_params = [p for n, p in named_parameters if include(n, p) and p.requires_grad]

    optimizer = optim.AdamW(
        [
            {"params": gain_or_bias_params, "weight_decay": 0.},
            {"params": rest_params, "weight_decay": args.wd},
        ],
        lr=args.lr,
        betas=(args.beta1, args.beta2),
        eps=args.eps,
    )

    scaler = GradScaler()

    start_epoch = 0
    data = get_data(args, (preprocess_train, preprocess_val), epoch=start_epoch, tokenizer=get_tokenizer(args.model))

    total_steps = (data["train"].dataloader.num_batches // args.accum_freq) * args.epochs
    scheduler = cosine_lr(optimizer, args.lr, args.warmup, total_steps)

    # evaluate(model, data, start_epoch, args)

    loss = ClipLoss(
            local_loss=args.local_loss,
            gather_with_grad=args.gather_with_grad,
            cache_labels=True,
            rank=args.rank,
            world_size=args.world_size,
            use_horovod=args.horovod)

    # setup debugger
    hook = None # set to none for non zero ranks
    if args.rank==0:
        
        # check if we're running on sagemaker
        if 'SM_RESOURCE_CONFIG' in os.environ:
            outdir = '/opt/ml/checkpoints/smdebugger'
        else:
            outdir = f'../smdebugger_{args.batch_size}'
        #decoder_input_re = "transformer_input_0"
        #decoder_re = "transformer\.h\.\d+\.attn\.attn_dropout_input_0"
        shutil.rmtree(outdir, ignore_errors=True)
        log_freq = args.log_every_n_steps
        #if Path(DEFAULT_CONFIG_FILE_PATH).exists():
            # smd.Hook.register_hook(pl_module.model, pl_module.criterion)
        #    hook = smd.Hook.create_from_json_file()
        #else:
        hook = smd.Hook(out_dir=outdir,
                        export_tensorboard=True,
                        reduction_config=ReductionConfig(reductions=['mean', 'isnan', 'isinf', 'std', 'min', 'max'], 
                                                         abs_reductions=['min','max'],
                                                         norms=['l1', 'l2']),
                        # reduction_config=ReductionConfig(reductions=['isnan'], norms=['l2']),
                        save_config=SaveConfig(save_interval=log_freq),
                        #include_regex="|".join([decoder_re, decoder_input_re]), #None,
                        include_collections=[CollectionKeys.LOSSES,
                                             CollectionKeys.GRADIENTS, 
                                             CollectionKeys.WEIGHTS],
                        save_all=False,
                        include_workers="one")
        hook.register_module(model)
        hook.register_loss(loss)

    
    for epoch in range(start_epoch, args.epochs):
        train_one_epoch(model, data, epoch, optimizer, scaler, scheduler, args, loss=loss, debugger_hook=hook)

    #evaluate(model, data, args.epochs, args)
    
if __name__=='__main__':
    cmdline = parse_args()
    cmdline_args, unknown_args = cmdline.parse_known_args()
    arg_dict.update(vars(cmdline_args))
    args = Namespace(**arg_dict)
    local_rank, rank, world_size = world_info_from_env()
    print(f"\n\nRank {rank} local rank {local_rank} of {world_size}\n\n")
    if rank==0:
        print(args)
    main(args)
    
import sys
sys.path.append('./src/')

import logging
import os
import sys
import random
from datetime import datetime
import shutil
from pathlib import Path

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
    "epochs": 50,
    "lr": 5e-4,
    "wd": 0.2,
    "eps": 1.0e-6,
    "beta1": 0.9,
    "beta2": 0.98,
    "precision": "amp_bfloat16",
    "workers": 4,
    "model": "ViT-L-14",
    "name": "proto_1",
    "pretrained": None, #"laion400m_e32",
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

args = Namespace(**arg_dict)

if torch.cuda.is_available():
    # This enables tf32 on Ampere GPUs which is only 8% slower than
    # float16 and almost as accurate as float32
    # This was a default in pytorch until 1.12
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
#if args.rank==0:
    
outdir = './smdebugger_1'
#decoder_input_re = "transformer_input_0"
#decoder_re = "transformer\.h\.\d+\.attn\.attn_dropout_input_0"
shutil.rmtree(outdir, ignore_errors=True)
log_freq = args.log_every_n_steps
if Path(DEFAULT_CONFIG_FILE_PATH).exists():
    # smd.Hook.register_hook(pl_module.model, pl_module.criterion)
    hook = smd.Hook.create_from_json_file()
else:
    hook = smd.Hook(out_dir=outdir,
                    export_tensorboard=True,
                    reduction_config=ReductionConfig(reductions=['mean', 'isnan', 'isinf', 'std', 'min', 'max'], norms=['l1', 'l2']),
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
    train_one_epoch(model, data, epoch, optimizer, scaler, scheduler, args, loss=loss)

evaluate(model, data, args.epochs, args)
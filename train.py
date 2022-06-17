import argparse
import collections
import logging
import math
import os
import random
import time

import numpy as np

import torch
import torch.distributed as distrib
import torch.nn.functional as F
import tqdm

from poni.dataset import SemanticMapPrecomputedDataset as SMPrecompDataset
from poni.default import get_cfg

from poni.model import get_semantic_encoder_decoder
from poni.train_utils import collate_fn, get_activation_fn, get_loss_fn
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


class SemanticMapperModule(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        # Define loss functions
        object_loss_type = self.cfg.MODEL.object_loss_type
        assert object_loss_type in ["bce", "l1", "l2", "xent"]
        self.object_loss_fn = get_loss_fn(object_loss_type)
        area_loss_type = self.cfg.MODEL.area_loss_type
        assert area_loss_type in ["bce", "l1", "l2"]
        self.area_loss_fn = get_loss_fn(area_loss_type)
        # Define models
        ndirs = None
        self.dirs_map = None
        self.inv_dists_map = None
        enable_directions = self.cfg.DATASET.enable_directions
        ndirs = len(self.cfg.DATASET.prediction_directions)
        enable_locations = self.cfg.DATASET.enable_locations
        enable_actions = self.cfg.DATASET.enable_actions
        assert not (enable_locations and enable_directions)
        assert not (enable_actions and enable_directions)
        assert not (enable_actions and enable_locations)
        enable_area_head = self.cfg.DATASET.enable_unexp_area
        if enable_directions:
            assert object_loss_type == "xent"
            assert self.cfg.MODEL.object_activation == "none"
            self.cfg.defrost()
            self.cfg.MODEL.output_type = "dirs"
            self.cfg.MODEL.ndirs = ndirs
            self.cfg.freeze()
        if enable_locations:
            assert object_loss_type in ["l1", "l2"]
            assert self.cfg.MODEL.object_activation == "sigmoid"
            self.cfg.defrost()
            self.cfg.MODEL.output_type = "locs"
            self.cfg.freeze()
        if enable_actions:
            assert object_loss_type == "xent"
            assert self.cfg.MODEL.object_activation == "none"
            self.cfg.defrost()
            self.cfg.MODEL.output_type = "acts"
            self.cfg.freeze()
        if enable_area_head:
            self.cfg.defrost()
            self.cfg.MODEL.enable_area_head = enable_area_head
            self.cfg.freeze()
        (
            self.encoder,
            self.object_decoder,
            self.area_decoder,
        ) = get_semantic_encoder_decoder(self.cfg)
        # Define optimizer
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.cfg.OPTIM.lr)
        # Define scheduler
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
            self.optimizer,
            milestones=cfg.OPTIM.lr_sched_milestones,
            gamma=cfg.OPTIM.lr_sched_gamma,
        )
        # Define activation functions
        self.object_activation = get_activation_fn(self.cfg.MODEL.object_activation)
        self.area_activation = get_activation_fn(self.cfg.MODEL.area_activation)

    def forward(self, x):
        embedding = self.encoder(x)
        object_preds = self.object_activation(self.object_decoder(embedding))
        area_preds = None
        if self.area_decoder is not None:
            area_preds = self.area_activation(self.area_decoder(embedding))
        return object_preds, area_preds

    def get_inv_dists_map(self, x):
        # x - (bs, N, M, M)
        M = x.shape[2]
        assert x.shape[3] == M
        Mby2 = M // 2
        # Compute a directions map that stores the direction value in
        # degrees for each location on the map.
        x_values = torch.arange(0, M, 1).float().unsqueeze(0) - Mby2  # (1, M)
        y_values = torch.arange(0, M, 1).float().unsqueeze(1) - Mby2  # (M, 1)
        dirs_map = torch.atan2(y_values, x_values)  # (M, M)
        dirs_map = torch.rad2deg(dirs_map).view(1, 1, M, M)
        dirs_map = (dirs_map + 360) % 360
        # Compute a distances map that stores the distance value in unit
        # cells for each location on the map.
        dists_map = torch.sqrt(x_values**2 + y_values**2)  # (M, M)
        inv_dists_map = torch.exp(-dists_map).view(1, 1, M, M)
        return dirs_map.to(x.device), inv_dists_map.to(x.device)

    def infer(self, x, do_forward_pass=True, input_maps=None, avg_preds=True):
        if do_forward_pass:
            object_preds, area_preds = self(x)
        else:
            assert input_maps is not None
            object_preds, area_preds = x
            x = input_maps

        if self.cfg.MODEL.output_type == "dirs":
            ####################################################################
            # Convert predicted directions to points on a map.
            ####################################################################
            # Convert predictions to angles
            # object_preds - (B, N, D)
            angles = torch.Tensor(self.cfg.DATASET.prediction_directions)
            angles = angles.to(object_preds.device)  # (D, )
            pred_dir_ixs = torch.argmax(object_preds, dim=2)  # (B, N)
            B, N = pred_dir_ixs.shape
            pred_dir_ixs = pred_dir_ixs.view(-1)  # (B * N)
            pred_dirs = torch.gather(angles, 0, pred_dir_ixs)
            pred_dirs = pred_dirs.view(B, N, 1, 1)  # (B, N, 1, 1)
            # Identify frontiers
            frontiers = self.calculate_frontiers(x)  # (B, 1, H, W)
            # Select the nearest frontier point along the predicted direction
            if self.dirs_map is None:
                self.dirs_map, self.inv_dists_map = self.get_inv_dists_map(x)
            dirs_map = self.dirs_map
            inv_dists_map = self.inv_dists_map
            delta = torch.abs(angles[1] - angles[0]).item() / 2
            dirs_map = dirs_map.to(x.device)
            abs_diff = torch.abs(dirs_map - pred_dirs) % 360  # (B, N, H, W)
            diff = (abs_diff - 180) % 360 - 180  # Convert from (0, 360) to (-180, 180)
            is_within_angle = (torch.abs(diff) <= delta).float()
            inv_dists_map = inv_dists_map.to(x.device)  # (1, 1, H, W)
            complex_mask = frontiers * is_within_angle * inv_dists_map  # (B, N, H, W)
            _, _, H, W = complex_mask.shape
            fpoint = torch.argmax(complex_mask.view(B, N, -1), dim=2)  # (B, N)
            # If no frontier point exists, sample a random point
            # along the predicted direction.
            no_frontier_mask = torch.all(
                complex_mask.view(B, N, -1) == 0, dim=2
            )  # (B, N)
            angle_mask = is_within_angle.view(B * N, H * W)
            spoint = torch.multinomial(angle_mask, 1)  # (B * N, 1)
            spoint = spoint.view(B, N)
            fpoint[no_frontier_mask] = spoint[no_frontier_mask]
            # Create a map with these predictions
            preds_map = torch.zeros_like(complex_mask)  # (B, N, H, W)
            preds_map = preds_map.view(B, N, -1)  # (B, N, H * W)
            preds_map.scatter_(2, fpoint.unsqueeze(2), 1.0)
            preds_map = preds_map.view(B, N, H, W)
            object_preds = F.max_pool2d(preds_map, 7, stride=1, padding=3)
        elif self.cfg.MODEL.output_type == "locs":
            ####################################################################
            # Convert predicted locations to points on a map.
            ####################################################################
            # Convert predictions to map locations
            # preds - (B, N, 2)
            B, N, H, W = x.shape
            preds_x = torch.clamp(object_preds[:, :, 0] * W, 0, W - 1)  # (B, N)
            preds_y = torch.clamp(object_preds[:, :, 1] * H, 0, H - 1)  # (B, N)
            # Convert to row-major form
            preds_xy = (preds_y * W + preds_x).long()  # (B, N)
            # Create a map with these predictions
            preds_map = torch.zeros_like(x)  # (B, N, H, W)
            preds_map = preds_map.view(B, N, -1)
            preds_map.scatter_(2, preds_xy.unsqueeze(2), 1.0)
            preds_map = preds_map.view(B, N, H, W)
            object_preds = F.max_pool2d(preds_map, 7, stride=1, padding=3)
        elif self.cfg.MODEL.output_type == "acts":
            ####################################################################
            # Retain predicted actions as actions
            ####################################################################
            # object_preds - (B, N, 4)
            assert not avg_preds
        # By default, average the two predictions and return it.
        if avg_preds:
            outputs = object_preds
            if area_preds is not None:
                outputs = (object_preds + area_preds) / 2.0
            return outputs
        else:
            return object_preds, area_preds

    def calculate_frontiers(self, x):
        # x - semantic map of shape (B, N, H, W)
        free_map = (x[:, 0] >= 0.5).float()  # (B, H, W)
        exp_map = torch.max(x, dim=1).values >= 0.5  # (B, H, W)
        unk_map = (~exp_map).float()  # (B, H, W)
        # Compute frontiers (reference below)
        # https://github.com/facebookresearch/exploring_exploration/blob/09d3f9b8703162fcc0974989e60f8cd5b47d4d39/exploring_exploration/models/frontier_agent.py#L132
        unk_map_shiftup = F.pad(unk_map, (0, 0, 0, 1))[:, 1:]
        unk_map_shiftdown = F.pad(unk_map, (0, 0, 1, 0))[:, :-1]
        unk_map_shiftleft = F.pad(unk_map, (0, 1, 0, 0))[:, :, 1:]
        unk_map_shiftright = F.pad(unk_map, (1, 0, 0, 0))[:, :, :-1]
        frontiers = (
            (free_map == unk_map_shiftup)
            | (free_map == unk_map_shiftdown)
            | (free_map == unk_map_shiftleft)
            | (free_map == unk_map_shiftright)
        ) & (
            free_map == 1
        )  # (B, H, W)
        # Dilate the frontiers
        frontiers = frontiers.unsqueeze(1).float()  # (B, 1, H, W)
        frontiers = torch.nn.functional.max_pool2d(frontiers, 7, stride=1, padding=3)
        return frontiers

    def undo_memory_opts(self, batch):
        inputs, labels = batch
        inputs["semmap"] = inputs["semmap"].float()
        labels["semmap"] = labels["semmap"].float()
        labels["object_pfs"] = labels["object_pfs"].float() / 1000.0
        if "area_pfs" in labels:
            labels["area_pfs"] = labels["area_pfs"].float() / 1000.0
        return (inputs, labels)

    def batch_step(self, batch):
        inputs, labels = batch
        input_maps = inputs["semmap"]
        object_preds, area_preds = self(input_maps)
        losses = {}
        if self.cfg.MODEL.output_type == "map":
            # object_preds - (B, N + 2, H, W)
            # Ignore free-space, wall predictions
            y_hat = object_preds[:, 2:]
            y = labels["object_pfs"][:, 2:]
            mask = labels["loss_masks"][:, 2:]
            loss = self.object_loss_fn(y_hat, y)
            # Evaluate predictions only on mask = 1 regions
            mask_sum = mask.sum(dim=3).sum(dim=2) + 1e-16  # (b, 1)
            loss = (loss * mask).sum(dim=3).sum(dim=2) / mask_sum  # (b, c)
            loss = loss.mean()
            losses["object_pf_loss"] = loss.item()
        elif self.cfg.MODEL.output_type == "dirs":
            D = len(self.cfg.DATASET.prediction_directions)
            # object_preds - (B, N + 2, D)
            # Ignore free-space, wall predictions
            y_hat = object_preds[:, 2:]  # (B, N, D)
            y = labels["dirs"][:, 2:]  # (B, N)
            # Replace non-object labels with 0 and mask these in the loss
            mask = y != D  # (B, N)
            y[~mask] = 0
            mask = mask.float()
            mask_sum = mask.sum(dim=1) + 1e-10  # (B, )
            y_hat = y_hat.permute(0, 2, 1)  # (B, D, N)
            loss = self.object_loss_fn(y_hat, y)  # (B, N)
            loss = (loss * mask).sum(dim=1) / mask_sum
            loss = loss.mean()
            losses["object_pf_loss"] = loss.item()
        elif self.cfg.MODEL.output_type == "locs":
            # object_preds - (B, N + 2, 2)
            # Ignore free-space, wall predictions
            y_hat = object_preds[:, 2:]  # (B, N, 2)
            y = labels["locs"][:, 2:]  # (B, N, 2)
            # Replace non-object labels with 0 and mask these in the loss
            mask = torch.all(y >= 0, dim=2, keepdim=True)  # (B, N, 1)
            mask = mask.expand(-1, -1, 2)  # (B, N, 2)
            y[~mask] = 0
            mask = mask.float()
            mask_sum = mask.sum(dim=2).sum(dim=1) + 1e-10  # (B, )
            loss = self.object_loss_fn(y_hat, y)  # (B, N, 2)
            loss = (loss * mask).sum(dim=2).sum(dim=1) / mask_sum  # (B, )
            loss = loss.mean()
            losses["object_pf_loss"] = loss.item()
        elif self.cfg.MODEL.output_type == "acts":
            # object_preds - (B, N + 2, 4)
            # Ignore free-space, wall predictions
            y_hat = object_preds[:, 2:]  # (B, N, 4)
            y = labels["acts"][:, 2:]  # (B, N)
            # Replace non-object labels with 0 and mask these in the loss
            mask = y >= 0  # (B, N)
            y[~mask] = 0
            mask = mask.float()
            mask_sum = mask.sum(dim=1) + 1e-10  # (B, )
            y_hat = y_hat.permute(0, 2, 1)  # (B, 4, N)
            loss = self.object_loss_fn(y_hat, y)  # (B, N)
            loss = (loss * mask).sum(dim=1) / mask_sum  # (B, )
            loss = loss.mean()
            losses["object_pf_loss"] = loss.item()

        if area_preds is not None:
            area_gts = labels["area_pfs"]  # (N, 1, H, W)
            area_pf_loss = self.area_loss_fn(area_preds, area_gts).mean()
            loss = loss + area_pf_loss
            losses["area_pf_loss"] = area_pf_loss.item()

        return {"loss": loss, "losses": losses}

    def train_dataloader(self, is_distributed=False):
        self.train_dataset = SMPrecompDataset(self.cfg.DATASET, split="train")
        self.train_sampler = None
        if is_distributed:
            self.train_sampler = torch.utils.data.distributed.DistributedSampler(
                self.train_dataset
            )
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.cfg.OPTIM.batch_size,
            shuffle=(self.train_sampler is None),
            num_workers=self.cfg.OPTIM.num_workers,
            pin_memory=True,
            sampler=self.train_sampler,
            collate_fn=collate_fn,
        )
        return self.train_loader

    def val_dataloader(self):
        self.val_dataset = SMPrecompDataset(self.cfg.DATASET, split="val")
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.cfg.OPTIM.batch_size,
            num_workers=self.cfg.OPTIM.num_workers,
            pin_memory=True,
            collate_fn=collate_fn,
        )
        return self.val_loader

    def test_dataloader(self):
        return self.val_dataloader()

    def update(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def convert_to_data_parallel(self):
        self.encoder = nn.DataParallel(self.encoder)
        self.object_decoder = nn.DataParallel(self.object_decoder)
        if self.area_decoder is not None:
            self.area_decoder = nn.DataParallel(self.area_decoder)

    def convert_to_distributed_data_parallel(self, rank):
        if self.cfg.LOGGING.verbose:
            print(f"=======> (0.3) breakpoint reached in local proc: {rank}")
        encoder = nn.SyncBatchNorm.convert_sync_batchnorm(self.encoder)
        object_decoder = nn.SyncBatchNorm.convert_sync_batchnorm(self.object_decoder)
        if self.area_decoder is not None:
            area_decoder = nn.SyncBatchNorm.convert_sync_batchnorm(self.area_decoder)
        if self.cfg.LOGGING.verbose:
            print(f"=======> (0.4) breakpoint reached in local proc: {rank}")
        self.encoder = DDP(encoder, device_ids=[rank])
        self.object_decoder = DDP(object_decoder, device_ids=[rank])
        if self.area_decoder is not None:
            self.area_decoder = DDP(area_decoder, device_ids=[rank])
        if self.cfg.LOGGING.verbose:
            print(f"=======> (0.5) breakpoint reached in local proc: {rank}")
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.cfg.OPTIM.lr)
        # Define scheduler
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
            self.optimizer,
            milestones=self.cfg.OPTIM.lr_sched_milestones,
            gamma=self.cfg.OPTIM.lr_sched_gamma,
        )

    def convert_object_pf_to_distance(self, opfs, min_value=1e-20, max_value=1.0):
        """
        opfs - (bs, N, H, W)
        """
        opfs = torch.clamp(opfs, min_value, max_value)
        data_cfg = self.cfg.DATASET
        max_d = data_cfg.object_pf_cutoff_dist
        dists = max_d - opfs * max_d
        return dists

    def convert_distance_to_object_pf(self, dists):
        """
        dists - (bs, N, H, W)
        """
        data_cfg = self.cfg.DATASET
        max_d = data_cfg.object_pf_cutoff_dist
        opfs = torch.clamp((max_d - dists) / max_d, 0.0, 1.0)
        return opfs

    def get_pf_cfg(self):
        return {"dthresh": self.cfg.DATASET.object_pf_cutoff_dist}


class Trainer:
    DEFAULT_PORT = 8738
    DEFAULT_PORT_RANGE = 127
    DEFAULT_MAIN_ADDR = "127.0.0.1"

    def __init__(self, cfg):
        self.cfg = cfg

        self.is_distributed = self.get_distrib_size()[2] > 1

        # Setup DDP
        local_rank, world_rank, world_size = 0, 0, 1
        if self.is_distributed:
            local_rank, world_rank, world_size = self.ddp_setup()
        self.rank = world_rank
        self.world_size = world_size

        # Seed everything
        random.seed(cfg.SEED + world_rank)
        np.random.seed(cfg.SEED + world_rank)
        torch.manual_seed(cfg.SEED + world_rank)
        torch.cuda.manual_seed_all(cfg.SEED + world_rank)

        # Create device
        self.device = torch.device(f"cuda:{local_rank}")
        # Create model
        self.model = SemanticMapperModule(cfg)
        self.model.to(self.device)
        if self.is_distributed:
            self.model.convert_to_distributed_data_parallel(local_rank)
        else:
            self.model.convert_to_data_parallel()

    def get_distrib_size(self):
        # Check to see if we should parse from torch.distributed.launch
        if os.environ.get("LOCAL_RANK", None) is not None:
            local_rank = int(os.environ["LOCAL_RANK"])
            world_rank = int(os.environ["RANK"])
            world_size = int(os.environ["WORLD_SIZE"])
        # Else parse from SLURM is using SLURM
        elif os.environ.get("SLURM_JOBID", None) is not None:
            local_rank = int(os.environ["SLURM_LOCALID"])
            world_rank = int(os.environ["SLURM_PROCID"])
            world_size = int(os.environ["SLURM_NTASKS"])
        # Otherwise setup for just 1 process, this is nice for testing
        else:
            local_rank = 0
            world_rank = 0
            world_size = 1

        return local_rank, world_rank, world_size

    def ddp_setup(self):
        assert distrib.is_available(), "torch.distributed must be available"

        local_rank, world_rank, world_size = self.get_distrib_size()

        slurm_jobid = os.environ.get("SLURM_JOB_ID", None)

        main_port = int(os.environ.get("MAIN_PORT", self.DEFAULT_PORT))
        if slurm_jobid is not None:
            main_port += int(slurm_jobid) % int(
                os.environ.get("MAIN_PORT_RANGE", self.DEFAULT_PORT_RANGE)
            )
        main_addr = os.environ.get("MAIN_ADDR", self.DEFAULT_MAIN_ADDR)
        if self.cfg.LOGGING.verbose:
            print(f"=======> (0) breakpoint reached in proc: {world_rank}")

        tcp_store = distrib.TCPStore(main_addr, main_port, world_size, world_rank == 0)
        if self.cfg.LOGGING.verbose:
            print(f"=======> (1) breakpoint reached in proc: {world_rank}")

        distrib.init_process_group(
            "nccl", store=tcp_store, rank=world_rank, world_size=world_size
        )
        if self.cfg.LOGGING.verbose:
            print(f"=======> (2) breakpoint reached in proc: {world_rank}")

        return local_rank, world_rank, world_size

    def ddp_cleanup(self):
        distrib.destroy_process_group()

    def train(self):
        cfg = self.cfg
        if self.rank == 0:
            # Setup loggers
            tb_dir = os.path.join(cfg.LOGGING.tb_dir, "train")
            self.tb_writer = SummaryWriter(log_dir=tb_dir)
            os.makedirs(cfg.LOGGING.log_dir, exist_ok=True)
            log_path = os.path.join(cfg.LOGGING.log_dir, "train.log")
            logger = logging.getLogger("train_pf")
            logger.setLevel(logging.DEBUG)
            vlog = logging.FileHandler(log_path)
            vlog.setLevel(logging.INFO)
            logger.addHandler(vlog)
            console = logging.StreamHandler()
            console.setLevel(logging.DEBUG)
            logger.addHandler(console)

        # Create data loaders
        train_loader = self.model.train_dataloader(self.is_distributed)
        val_loader = self.model.val_dataloader()
        n_train_samples = len(self.model.train_dataset)

        if self.rank == 0:
            logger.info(f"=======> Total world size: {self.world_size}")
            os.makedirs(cfg.LOGGING.ckpt_dir, exist_ok=True)

        # Initialize weights from pretrained model
        if cfg.MODEL.pretrained_path != "":
            print(f"===> Initializing weights from {cfg.MODEL.pretrained_path}")
            ckpt = torch.load(cfg.MODEL.pretrained_path)
            self.model.load_state_dict(ckpt["state_dict"])

        # Resume checkpoint if available
        ckpt_path = os.path.join(cfg.LOGGING.ckpt_dir, "last.ckpt")
        start_epoch = 0
        start_steps = 0
        start_updates = 0
        best_val_loss = math.inf
        if os.path.isfile(ckpt_path):
            if self.rank == 0:
                logger.info(f"====> Resuming checkpoint from {ckpt_path}")
            ckpt = torch.load(ckpt_path)
            self.model.load_state_dict(ckpt["state_dict"])
            self.model.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            if "lr_scheduler_state_dict" in ckpt:
                self.model.scheduler.load_state_dict(ckpt["lr_scheduler_state_dict"])
            start_epoch = ckpt["epoch"] + 1
            start_steps = ckpt["running_steps"] + 1
            start_updates = ckpt["num_updates"] + 1
            best_val_loss = ckpt["best_val_loss"]

        running_steps = start_steps

        # Wait for other processes to also resume checkpoint
        if self.is_distributed > 1:
            distrib.barrier()

        # Start training
        train_losses = collections.deque(maxlen=200)
        train_sep_losses = collections.defaultdict(
            lambda: collections.deque(maxlen=200)
        )
        eff_batch_size = cfg.OPTIM.batch_size * self.world_size
        updates_per_epoch = math.ceil(n_train_samples / eff_batch_size)
        num_epochs = math.ceil(cfg.OPTIM.num_total_updates / updates_per_epoch)
        num_updates = start_updates
        if self.rank == 0:
            logger.info(f"Effective batch size: {eff_batch_size}")
            logger.info(f"# updates per epoch: {updates_per_epoch}")
            logger.info(f"# epochs to train: {num_epochs}")
        for epoch in range(start_epoch, num_epochs):
            if self.is_distributed:
                self.model.train_sampler.set_epoch(epoch)
            ############################# training loop ########################
            self.model.train()
            # -----------------------------------------------------------------#
            dt_start_time = time.time()
            step = 0
            pt_time = 0.0  # pytorch time
            dt_time = 0.0  # data time
            for batch_id, (inputs, labels) in enumerate(train_loader):
                num_updates += 1
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                curr_step = inputs[list(inputs.keys())[0]].shape[0]
                for key in [
                    "object_pfs",
                    "loss_masks",
                    "dirs",
                    "locs",
                    "area_pfs",
                    "acts",
                ]:
                    if key in labels and labels[key] is not None:
                        labels[key] = labels[key].to(self.device)
                curr_dt_time = time.time() - dt_start_time
                # -------------------------------------------------------------#
                pt_start_time = time.time()
                # Forward pass
                batch = (inputs, labels)
                batch = self.model.undo_memory_opts(batch)
                outputs = self.model.batch_step(batch)
                loss = outputs["loss"]
                # Backward pass
                self.model.update(loss)
                train_losses.append(loss.item())
                for k, v in outputs["losses"].items():
                    train_sep_losses[k].append(v)
                curr_pt_time = time.time() - pt_start_time
                # ---------------- sum statistics across processes ------------#
                if self.is_distributed:
                    curr_stats = torch.Tensor(
                        [curr_step, curr_dt_time, curr_pt_time]
                    ).to(self.device)
                    distrib.all_reduce(curr_stats)
                    curr_step, curr_dt_time, curr_pt_time = (
                        curr_stats.cpu().numpy().tolist()
                    )
                # --------------------- update statistics ---------------------#
                dt_time += curr_dt_time
                pt_time += curr_pt_time
                step += int(curr_step)
                running_steps += int(curr_step)
                # ---------------------- logging metrics ----------------------#
                if (num_updates % cfg.LOGGING.log_interval == 0) and (self.rank == 0):
                    avg_train_loss = np.mean(train_losses).item()
                    lr = self.model.optimizer.param_groups[0]["lr"]
                    self.tb_writer.add_scalar("LR", lr, running_steps)
                    self.tb_writer.add_scalar(
                        "train/loss", avg_train_loss, running_steps
                    )
                    self.tb_writer.add_scalar("train/epochs", epoch, running_steps)
                    for k, v in train_sep_losses.items():
                        avg_v = np.mean(v).item()
                        self.tb_writer.add_scalar(f"train/{k}", avg_v, running_steps)
                    avg_pt_time = pt_time / (batch_id + 1) / 60.0  # minutes
                    avg_dt_time = dt_time / (batch_id + 1) / 60.0  # minutes
                    self.tb_writer.add_scalar(
                        "time/pytorch", avg_pt_time, running_steps
                    )
                    self.tb_writer.add_scalar("time/data", avg_dt_time, running_steps)
                    logger.info("=" * 30)
                    logger.info(
                        f" Epoch [{epoch:4d}/{num_epochs:4d}] |"
                        f" Step [{step:6d}/{n_train_samples:6d}] |"
                        f" Train loss: {avg_train_loss:8.4f} |"
                        f" LR: {lr:8.4f} |"
                        f" PTime (min): {avg_pt_time:6.2f} |"
                        f" DTime (min): {avg_dt_time:6.2f}"
                    )
                    print("======> Complete losses")
                    for k, v in train_sep_losses.items():
                        avg_v = np.mean(v).item()
                        logger.info(f"{k:<15s} | {avg_v:8.4f}")

                ######################### evaluation loop ######################
                if num_updates % cfg.LOGGING.eval_interval == 0:
                    val_losses = []
                    val_sep_losses = collections.defaultdict(list)
                    self.model.eval()
                    val_size = len(self.model.val_dataset)
                    total_batches = val_size // self.cfg.OPTIM.batch_size
                    if total_batches * self.cfg.OPTIM.batch_size < val_size:
                        total_batches += 1
                    # ---------------------------------------------------------#
                    for inputs, labels in tqdm.tqdm(val_loader, total=total_batches):
                        inputs = {k: v.to(self.device) for k, v in inputs.items()}
                        for key in [
                            "object_pfs",
                            "loss_masks",
                            "dirs",
                            "locs",
                            "area_pfs",
                            "acts",
                        ]:
                            if key in labels and labels[key] is not None:
                                labels[key] = labels[key].to(self.device)
                        # -----------------------------------------------------#
                        batch = (inputs, labels)
                        batch = self.model.undo_memory_opts(batch)
                        with torch.no_grad():
                            outputs = self.model.batch_step(batch)
                        loss = outputs["loss"]
                        val_losses.append(loss.item())
                        for k, v in outputs["losses"].items():
                            val_sep_losses[k].append(v)
                    self.model.train()
                    val_losses = np.mean(val_losses).item()
                    # -------------------- logging metrics --------------------#
                    if self.rank == 0:
                        self.tb_writer.add_scalar("val/loss", val_losses, running_steps)
                        for k, v in val_sep_losses.items():
                            avg_v = np.mean(v).item()
                            self.tb_writer.add_scalar(f"val/{k}", avg_v, running_steps)

                        logger.info(f"===========> Val loss: {val_losses:8.4f}")
                        logger.info("======> Complete val losses")
                        for k, v in val_sep_losses.items():
                            avg_v = np.mean(v).item()
                            logger.info(f"{k:<15s} | {avg_v:8.4f}")

                ######################### save checkpoint ######################
                if num_updates % cfg.LOGGING.ckpt_interval == 0 and self.rank == 0:
                    ckpt = {
                        "state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.model.optimizer.state_dict(),
                        "lr_scheduler_state_dict": self.model.scheduler.state_dict(),
                        "epoch": epoch,
                        "running_steps": running_steps,
                        "cfg": cfg,
                        "best_val_loss": best_val_loss,
                        "num_updates": num_updates,
                    }
                    logger.info("============> Saving checkpoint")

                    # Save best checkpoint
                    if best_val_loss > val_losses:
                        best_val_loss = val_losses
                        ckpt["best_val_loss"] = best_val_loss
                        ckpt_path = os.path.join(cfg.LOGGING.ckpt_dir, "best.ckpt")
                        torch.save(ckpt, ckpt_path)

                    # Save latest checkpoint
                    ckpt_path = os.path.join(cfg.LOGGING.ckpt_dir, "last.ckpt")
                    torch.save(ckpt, ckpt_path)

                # --------------------------------------------------------------#
                dt_start_time = time.time()
                if self.is_distributed > 1:
                    # The other processes wait for rank 0 process to finish
                    # validating and saving checkpoints.
                    distrib.barrier()

            self.model.scheduler.step()

        # Cleanup after training
        self.ddp_cleanup()


def main(exp_config, opts=None) -> None:
    cfg = get_cfg(exp_config, opts)
    trainer = Trainer(cfg)
    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-config", type=str, default="")
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)

    args = parser.parse_args()

    main(args.exp_config, args.opts)

import os
import random
import string
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import hydra
import omegaconf
import pytorch_lightning as pl
import torch
import torch.multiprocessing
from omegaconf import OmegaConf, listconfig
from pytorch_lightning import LightningModule
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.utilities import rank_zero_only

from boltz.data.module.training import BoltzTrainingDataModule, DataConfig


def get_teacher_model(model_name: str, checkpoint_path: str) -> LightningModule:
    if model_name == "boltz1":
        from boltz.model.models.boltz1 import Boltz1

        pairformer_args = dict(num_blocks=48, num_heads=16, dropout=0.25,
                               activation_checkpointing=False, # No need because no optimization
                               offload_to_cpu=False)

        predict_args = {
            "recycling_steps": 0, # This is ignored
            "sampling_steps": 200,
            "diffusion_samples": 1,
            "max_parallel_samples": 1,
            "write_confidence_summary": False,
            "write_full_pae": False,
            "write_full_pde": False,
        }

        diffusion_params = dict(gamma_0=0.605, gamma_min=1.107, noise_scale=0.901, rho=8, step_scale=1.638,
                                sigma_min=0.0004, sigma_max=160.0, sigma_data=16.0, P_mean=-1.2, P_std=1.5,
                                coordinate_augmentation=True, alignment_reverse_diff=True, synchronize_sigmas=True,
                                use_inference_model_cache=True)

        msa_args = dict(msa_s=64, msa_blocks=4, msa_dropout=0.0, z_dropout=0.0, use_paired_feature=False,
                        pairwise_head_width=32, pairwise_num_heads=4, activation_checkpointing=False,
                        offload_to_cpu=False, subsample_msa=False, num_subsampled_msa=1024)

        steering_args = dict(fk_steering=False, num_particles=3, fk_lambda=4.0, fk_resampling_interval=3,
                             physical_guidance_update=False, contact_guidance_update=True, num_gd_steps=20)

        model_module = Boltz1.load_from_checkpoint(
            checkpoint_path=checkpoint_path,
            strict=True,  # TODO: should this be false for loading without confidence
            predict_args=predict_args,
            map_location="cpu",
            diffusion_process_args=diffusion_params,
            ema=False,
            use_kernels=False, # I am not sure what this is, something with acceleration?
            pairformer_args=pairformer_args,
            msa_args=msa_args,
            steering_args=steering_args,
        )
        model_module.eval()
        return model_module

    else:
        raise ValueError(f"Unknown model name: {model_name}")


@dataclass
class TrainConfig:
    """Train configuration.

    Attributes
    ----------
    data : DataConfig
        The data configuration.
    model : ModelConfig
        The model configuration.
    output : str
        The output directory.
    trainer : Optional[dict]
        The trainer configuration.
    resume : Optional[str]
        The resume checkpoint.
    pretrained : Optional[str]
        The pretrained model.
    wandb : Optional[dict]
        The wandb configuration.
    disable_checkpoint : bool
        Disable checkpoint.
    matmul_precision : Optional[str]
        The matmul precision.
    find_unused_parameters : Optional[bool]
        Find unused parameters.
    save_top_k : Optional[int]
        Save top k checkpoints.
    validation_only : bool
        Run validation only.
    debug : bool
        Debug mode.
    strict_loading : bool
        Fail on mismatched checkpoint weights.
    load_confidence_from_trunk: Optional[bool]
        Load pre-trained confidence weights from trunk.

    """

    data: DataConfig
    model: LightningModule
    output: str
    trainer: Optional[dict] = None
    resume: Optional[str] = None
    pretrained: Optional[str] = None
    wandb: Optional[dict] = None
    disable_checkpoint: bool = False
    matmul_precision: Optional[str] = None
    find_unused_parameters: Optional[bool] = False
    save_top_k: Optional[int] = 1
    validation_only: bool = False
    debug: bool = False
    strict_loading: bool = True
    load_confidence_from_trunk: Optional[bool] = False
    teacher_model: Optional[dict] = None


def train(raw_config: str, args: list[str]) -> None:  # noqa: C901, PLR0912, PLR0915
    """Run training.

    Parameters
    ----------
    raw_config : str
        The input yaml configuration.
    args : list[str]
        Any command line overrides.

    """
    # Load the configuration
    raw_config = omegaconf.OmegaConf.load(raw_config)

    # Apply input arguments
    args = omegaconf.OmegaConf.from_dotlist(args)
    raw_config = omegaconf.OmegaConf.merge(raw_config, args)

    # Instantiate the task
    cfg = hydra.utils.instantiate(raw_config)
    cfg = TrainConfig(**cfg)

    # Set matmul precision
    if cfg.matmul_precision is not None:
        torch.set_float32_matmul_precision(cfg.matmul_precision)

    # Create trainer dict
    trainer = cfg.trainer
    if trainer is None:
        trainer = {}

    # Flip some arguments in debug mode
    devices = trainer.get("devices", 1)

    wandb = cfg.wandb
    if cfg.debug:
        if isinstance(devices, int):
            devices = 1
        elif isinstance(devices, (list, listconfig.ListConfig)):
            devices = [devices[0]]
        trainer["devices"] = devices
        cfg.data.num_workers = 0
        if wandb:
            wandb = None

    # Create objects
    data_config = DataConfig(**cfg.data)
    data_module = BoltzTrainingDataModule(data_config)
    model_module = cfg.model

    if cfg.pretrained and not cfg.resume:
        # Load the pretrained weights into the confidence module
        if cfg.load_confidence_from_trunk:
            checkpoint = torch.load(cfg.pretrained, map_location="cpu")

            # Modify parameter names in the state_dict
            new_state_dict = {}
            for key, value in checkpoint["state_dict"].items():
                if not key.startswith("structure_module") and not key.startswith(
                    "distogram_module"
                ):
                    new_key = "confidence_module." + key
                    new_state_dict[new_key] = value
            new_state_dict.update(checkpoint["state_dict"])

            # Update the checkpoint with the new state_dict
            checkpoint["state_dict"] = new_state_dict

            # Save the modified checkpoint
            random_string = "".join(
                random.choices(string.ascii_lowercase + string.digits, k=10)
            )
            file_path = os.path.dirname(cfg.pretrained) + "/" + random_string + ".ckpt"
            print(
                f"Saving modified checkpoint to {file_path} created by broadcasting trunk of {cfg.pretrained} to confidence module."
            )
            torch.save(checkpoint, file_path)
        else:
            file_path = cfg.pretrained

        print(f"Loading model from {file_path}")
        model_module = type(model_module).load_from_checkpoint(
            file_path, map_location="cpu", strict=False, **(model_module.hparams)
        )

        if cfg.load_confidence_from_trunk:
            os.remove(file_path)
    if cfg.teacher_model:
        model_module.teacher_model = get_teacher_model(**cfg.teacher_model)

    # Create checkpoint callback
    callbacks = []
    dirpath = cfg.output
    if not cfg.disable_checkpoint:
        mc = ModelCheckpoint(
            monitor="val/lddt",
            save_top_k=cfg.save_top_k,
            save_last=True,
            mode="max",
            every_n_epochs=1,
        )
        callbacks = [mc]

    # Create wandb logger
    loggers = []
    last_ckpt_path = None
    if wandb:
        run_id_file = Path(cfg.output) / "wandb_run_id.txt"

        @rank_zero_only
        def create_run_id():
            if not run_id_file.exists():
                new_run_id = "".join(random.choices(string.ascii_lowercase + string.digits, k=8))
                os.makedirs(run_id_file.parent, exist_ok=True)
                run_id_file.write_text(new_run_id)

        # Make sure all ranks see the same run_id
        create_run_id()
        while not run_id_file.exists():
            print(f"Rank {os.getenv('RANK')}: Waiting for run_id file to be created...")
            time.sleep(1)
        run_id = run_id_file.read_text().strip()

        last_ckpt_path = Path(cfg.output) / cfg.wandb["project"] / run_id / "checkpoints" / "last.ckpt"

        wdb_logger = WandbLogger(
            name=wandb["name"],
            group=wandb["name"],
            save_dir=cfg.output,
            project=wandb["project"],
            entity=wandb["entity"],
            log_model=False,
            id=run_id,          # ensures resuming the same run
            resume="allow",     # resume if run already exists
        )
        loggers.append(wdb_logger)
        # Save the config to wandb

        @rank_zero_only
        def save_config_to_wandb() -> None:
            config_out = Path(wdb_logger.experiment.dir) / "run.yaml"
            with Path.open(config_out, "w") as f:
                OmegaConf.save(raw_config, f)
            wdb_logger.experiment.save(str(config_out))

        save_config_to_wandb()

    # Set up trainer
    strategy = "auto"
    if (isinstance(devices, int) and devices > 1) or (
        isinstance(devices, (list, listconfig.ListConfig)) and len(devices) > 1
    ):
        strategy = DDPStrategy(find_unused_parameters=cfg.find_unused_parameters)

    if trainer["num_nodes"] == "auto":
        trainer["num_nodes"] = int(os.environ.get("SLURM_JOB_NUM_NODES", 1))

    trainer = pl.Trainer(
        default_root_dir=str(dirpath),
        strategy=strategy,
        callbacks=callbacks,
        logger=loggers,
        enable_checkpointing=not cfg.disable_checkpoint,
        reload_dataloaders_every_n_epochs=1,
        **trainer,
    )

    if not cfg.strict_loading:
        model_module.strict_loading = False

    ckpt_path = cfg.resume
    if last_ckpt_path and last_ckpt_path.exists():
        print(f"Resuming from last checkpoint at {last_ckpt_path}")
        ckpt_path = str(last_ckpt_path)

    if cfg.validation_only:
        trainer.validate(
            model_module,
            datamodule=data_module,
            ckpt_path=ckpt_path,
        )
    else:
        trainer.fit(
            model_module,
            datamodule=data_module,
            ckpt_path=ckpt_path,
        )


if __name__ == "__main__":
    arg1 = sys.argv[1]
    arg2 = sys.argv[2:]
    train(arg1, arg2)

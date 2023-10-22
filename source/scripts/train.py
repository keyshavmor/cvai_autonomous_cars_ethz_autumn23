import os
import shutil
import uuid

from datetime import datetime

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger, TestTubeLogger

from source.experiments import *
from source.utils.rules import check_all_rules, pack_submission
from source.utils.config import command_line_parser


def main():
    cfg = command_line_parser()
    TEAM_ID = int(os.environ.get("TEAM_ID", "0"))

    # Remove previous logs and check file structure
    if os.path.isdir(cfg.log_dir):
        shutil.rmtree(cfg.log_dir)
    check_all_rules(cfg)

    # Resolve name task
    experiment_name = "Experiment" + "".join(sorted([s.capitalize() for s in cfg.tasks]))
    model = eval(experiment_name)(cfg)

    timestamp = datetime.now().strftime('%m%d-%H%M')
    run_name = f'T{TEAM_ID:02d}_{timestamp}_{cfg.name}_{str(uuid.uuid4())[:5]}'

    tube_logger = TestTubeLogger(
        save_dir=cfg.log_dir,
        name='tube',
        version=0,
    )
    wandb_logger = WandbLogger(
        name=run_name,
        project='CVAIAC-Ex1',
        )

    checkpoint_local_callback = ModelCheckpoint(
        dirpath=os.path.join(cfg.log_dir, 'checkpoints'),
        save_last=False,
        save_top_k=1,
        monitor='metrics_summary/total',
        mode='max',
    )

    # Log information to wandb
    meta = {
        "Team_Id": TEAM_ID
    }
    wandb_logger.log_hyperparams(meta)

    trainer = Trainer(
        logger=[wandb_logger, tube_logger],
        callbacks=[checkpoint_local_callback],
        gpus='-1' if torch.cuda.is_available() else None,
        resume_from_checkpoint=cfg.resume,
        max_epochs=cfg.num_epochs,
        distributed_backend=None,
        weights_summary=None,
        weights_save_path=None,
        num_sanity_val_steps=1,
        precision=16 if cfg.optimizer_float_16 else 32,
        progress_bar_refresh_rate=100,
        log_every_n_steps=50,
        # Uncomment the following options if you want to try out framework changes without training too long
        # limit_train_batches=200,
        # limit_val_batches=10,
        # limit_test_batches=10,
        # log_every_n_steps=10,
    )

    if not cfg.prepare_submission:
        trainer.fit(model)

    # prepare submission archive with predictions, source code, training log, and the model
    dir_pred = os.path.join(cfg.log_dir, 'predictions')
    shutil.rmtree(dir_pred, ignore_errors=True)
    trainer.test(model)
    pack_submission(cfg.log_dir, submission_name=f"submission_{run_name}.zip")


if __name__ == '__main__':
    main()

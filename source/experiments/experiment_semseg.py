import os
import torch
import wandb
from torch.utils.data.dataloader import default_collate

from source.datasets.definitions import *
from source.utils.visualization import compose
from source.utils.helpers import resolve_model_class
from source.experiments.experiment_semseg_with_depth import ExperimentDepthSemseg


class ExperimentSemseg(ExperimentDepthSemseg):
    def __init__(self, cfg):
        super(ExperimentSemseg, self).__init__(cfg=cfg)

    def instantiate_model(self, cfg):
        self.outputs_descriptor = {
            MOD_SEMSEG: self.semseg_num_classes,
        }
        model_class = resolve_model_class(cfg.model_name)
        self.net = model_class(cfg, self.outputs_descriptor)
        print(self.net)

    def training_step(self, batch, batch_nb):
        rgb = batch[MOD_RGB]
        y_semseg_lbl = batch[MOD_SEMSEG].squeeze(1)

        if torch.cuda.is_available():
            rgb = rgb.cuda()
            y_semseg_lbl = y_semseg_lbl.cuda()

        y_hat = self.net(rgb)
        y_hat_semseg = y_hat[MOD_SEMSEG]

        if isinstance(y_hat_semseg, list):
            # deep supervision scenario: penalize all predicitons in the list and average losses
            loss_semseg = sum([self.loss_semseg(y_hat_semseg_i, y_semseg_lbl) for y_hat_semseg_i in y_hat_semseg])
            loss_semseg = loss_semseg / len(y_hat_semseg)
            y_hat_semseg = y_hat_semseg[-1]
        else:
            loss_semseg = self.loss_semseg(y_hat_semseg, y_semseg_lbl)

        loss_total = self.cfg.loss_weight_semseg * loss_semseg

        self.log_dict({
                'loss_train/semseg': loss_semseg,
                'loss_train/total': loss_total,
            }, on_step=True, on_epoch=False, prog_bar=True
        )

        if self.can_visualize():
            self.visualize(batch, y_hat_semseg, batch[MOD_ID], 'imgs_train/batch_crops')

        return {
            'loss': loss_total,
        }

    def inference_step(self, batch):
        rgb = batch[MOD_RGB]

        if torch.cuda.is_available():
            rgb = rgb.cuda()

        y_hat = self.net(rgb)
        y_hat_semseg = y_hat[MOD_SEMSEG]

        if isinstance(y_hat_semseg, list):
            y_hat_semseg = y_hat_semseg[-1]

        y_hat_semseg_lbl = y_hat_semseg.argmax(dim=1)

        return y_hat_semseg, y_hat_semseg_lbl

    def validation_step(self, batch, batch_nb):
        y_hat_semseg, y_hat_semseg_lbl = self.inference_step(batch)

        y_semseg_lbl = batch[MOD_SEMSEG].squeeze(1)

        if torch.cuda.is_available():
            y_semseg_lbl = y_semseg_lbl.cuda()

        loss_val_semseg = self.loss_semseg(y_hat_semseg, y_semseg_lbl)

        self.metrics_semseg.update_batch(y_hat_semseg_lbl, y_semseg_lbl)

        self.log_dict({
                'loss_val/semseg': loss_val_semseg,
                'loss_val/total': loss_val_semseg,
            }, on_step=False, on_epoch=True
        )

    def validation_epoch_end(self, outputs):
        self.observer_step()

        metrics_semseg = self.metrics_semseg.get_metrics_summary()
        self.metrics_semseg.reset()

        metric_semseg = (metrics_semseg['mean_iou'] - 50).clamp(min=0)

        scalar_logs = {
            'metrics_summary/semseg': metric_semseg,
            'metrics_summary/total': metric_semseg,
            'trainer/LR': torch.tensor(self.trainer.lr_schedulers[0]["scheduler"].get_last_lr()[0]),
        }
        scalar_logs.update({f'metrics_task_semseg/{k.replace(" ", "_")}': v for k, v in metrics_semseg.items()})

        self.log_dict(scalar_logs, on_step=False, on_epoch=True)

    def test_step(self, batch, batch_nb):
        _, y_hat_semseg_lbl = self.inference_step(batch)
        path_pred = os.path.join(self.cfg.log_dir, 'predictions')
        path_pred_semseg = os.path.join(path_pred, MOD_SEMSEG)
        if batch_nb == 0:
            os.makedirs(path_pred_semseg)
        split_test = SPLIT_TEST
        for i in range(y_hat_semseg_lbl.shape[0]):
            sample_name = self.datasets[split_test].name_from_index(batch[MOD_ID][i])
            path_file_semseg = os.path.join(path_pred_semseg, f'{sample_name}.png')
            pred_semseg = y_hat_semseg_lbl[i]
            self.datasets[split_test].save_semseg(
                path_file_semseg, pred_semseg, self.semseg_class_colors, self.semseg_ignore_label
            )

    def visualize(self, batch, y_hat_semseg, rgb_tags, tag):
        batch = {k: v.cpu().detach() for k, v in batch.items() if torch.is_tensor(v)}
        y_hat_semseg_lbl = y_hat_semseg.cpu().detach().argmax(dim=1)
        visualization_plan = [
            (MOD_RGB, batch[MOD_RGB], rgb_tags),
            (MOD_SEMSEG, batch[MOD_SEMSEG], 'GT SemSeg'),
            (MOD_SEMSEG, y_hat_semseg_lbl, 'Prediction SemSeg'),
        ]
        vis = compose(
            visualization_plan,
            self.cfg,
            rgb_mean=self.rgb_mean,
            rgb_stddev=self.rgb_stddev,
            semseg_color_map=self.semseg_class_colors,
            semseg_ignore_label=self.semseg_ignore_label,
        )
        # Use commit=False to not increment the step counter
        self.logger.experiment[0].log({
            tag: [wandb.Image(vis.cpu(), caption=tag)]
        }, commit=False)

    def can_visualize(self):
        return (not torch.cuda.is_available() or torch.cuda.current_device() == 0) and (
                self.global_step - self.cfg.num_steps_visualization_first) % \
                self.cfg.num_steps_visualization_interval == 0

    def observer_step(self):
        if torch.cuda.is_available() and torch.cuda.current_device() != 0:
            return
        vis_transforms = self.transforms_val_test
        list_samples = []
        for i in self.cfg.observe_train_ids:
            list_samples.append(self.datasets[SPLIT_TRAIN].get(i, override_transforms=vis_transforms))
        for i in self.cfg.observe_valid_ids:
            list_samples.append(self.datasets[SPLIT_VALID].get(i, override_transforms=vis_transforms))
        list_prefix = ('imgs_train/', ) * len(self.cfg.observe_train_ids) + ('imgs_val/', ) * len(self.cfg.observe_valid_ids)
        batch = default_collate(list_samples)
        rgb = batch[MOD_RGB]
        rgb_tags = [f'{prefix}{id}' for prefix, id in zip(list_prefix, batch[MOD_ID])]
        with torch.no_grad():
            if torch.cuda.is_available():
                rgb = rgb.cuda()
            y_hat = self.net(rgb)
            y_hat_semseg = y_hat[MOD_SEMSEG]
            if isinstance(y_hat_semseg, list):
                y_hat_semseg = y_hat_semseg[-1]
        self.visualize(batch, y_hat_semseg, rgb_tags, 'imgs_val/observed_samples')

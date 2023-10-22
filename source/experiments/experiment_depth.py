import os
import torch
import wandb
from torch.utils.data.dataloader import default_collate

from source.datasets.definitions import *
from source.utils.helpers import resolve_model_class
from source.utils.visualization import compose
from source.experiments.experiment_semseg_with_depth import ExperimentDepthSemseg


class ExperimentDepth(ExperimentDepthSemseg):
    def __init__(self, cfg):
        super(ExperimentDepth, self).__init__(cfg=cfg)

    def instantiate_model(self, cfg):
        self.outputs_descriptor = {
            MOD_DEPTH: 1,
        }
        model_class = resolve_model_class(cfg.model_name)
        self.net = model_class(cfg, self.outputs_descriptor)
        print(self.net)

    def training_step(self, batch, batch_nb):
        rgb = batch[MOD_RGB]
        y_depth = batch[MOD_DEPTH].squeeze(1)

        if torch.cuda.is_available():
            rgb = rgb.cuda()
            y_depth = y_depth.cuda()

        y_hat = self.net(rgb)
        y_hat_depth = y_hat[MOD_DEPTH]

        if type(y_hat_depth) is list:
            # deep supervision scenario: penalize all predicitons in the list and average losses
            loss_depth = sum([self.loss_depth(y_hat_depth_i, y_depth) for y_hat_depth_i in y_hat_depth])
            loss_depth = loss_depth / len(y_hat_depth)
            y_hat_depth = y_hat_depth[-1]
        else:
            loss_depth = self.loss_depth(y_hat_depth, y_depth)

        loss_total = self.cfg.loss_weight_depth * loss_depth

        self.log_dict({
                'loss_train/depth': loss_depth,
                'loss_train/total': loss_total,
            }, on_step=True, on_epoch=False, prog_bar=True
        )

        if self.can_visualize():
            self.visualize(batch, y_hat_depth, batch[MOD_ID], 'imgs_train/batch_crops')

        return {
            'loss': loss_total,
        }

    def inference_step(self, batch):
        rgb = batch[MOD_RGB]

        if torch.cuda.is_available():
            rgb = rgb.cuda()

        y_hat = self.net(rgb)
        y_hat_depth = y_hat[MOD_DEPTH]
        if isinstance(y_hat_depth, list):
            y_hat_depth = y_hat_depth[-1]

        y_hat_depth_normalized = (y_hat_depth - self.depth_meters_mean) / self.depth_meters_stddev 

        return y_hat_depth, y_hat_depth_normalized

    def validation_step(self, batch, batch_nb):
        y_hat_depth, _ = self.inference_step(batch)
        y_depth = batch[MOD_DEPTH].squeeze(1)

        if torch.cuda.is_available():
            y_depth = y_depth.cuda()

        loss_val_depth = self.loss_depth(y_hat_depth, y_depth)
        self.metrics_depth.update_batch(y_hat_depth, y_depth)

        self.log_dict({
                'loss_val/depth': loss_val_depth,
                'loss_val/total': loss_val_depth,
            }, on_step=False, on_epoch=True
        )

    def validation_epoch_end(self, outputs):
        self.observer_step()

        metrics_depth = self.metrics_depth.get_metrics_summary()
        self.metrics_depth.reset()

        metric_depth = (50 - metrics_depth['si_log_rmse']).clamp(min=0)

        scalar_logs = {
            'metrics_summary/depth': metric_depth,
            'metrics_summary/total': metric_depth,
            'trainer/LR': torch.tensor(self.trainer.lr_schedulers[0]["scheduler"].get_last_lr()[0]),
        }
        scalar_logs.update({f'metrics_task_depth/{k}': v for k, v in metrics_depth.items()})

        self.log_dict(scalar_logs, on_step=False, on_epoch=True)

    def test_step(self, batch, batch_nb):
        y_hat_depth, _ = self.inference_step(batch)
        path_pred = os.path.join(self.cfg.log_dir, 'predictions')
        path_pred_depth = os.path.join(path_pred, MOD_DEPTH)
        if batch_nb == 0:
            os.makedirs(path_pred_depth)
        split_test = SPLIT_TEST
        for i in range(y_hat_depth.shape[0]):
            sample_name = self.datasets[split_test].name_from_index(batch[MOD_ID][i])
            path_file_depth = os.path.join(path_pred_depth, f'{sample_name}.png')
            pred_depth = y_hat_depth[i]
            self.datasets[split_test].save_depth(
                path_file_depth, pred_depth, out_of_range_policy='clamp_to_range'
            )

    def visualize(self, batch, y_hat_depth, rgb_tags, tag):
        batch = {k: v.cpu().detach() for k, v in batch.items() if torch.is_tensor(v)}
        y_hat_depth = y_hat_depth.cpu().detach()
        visualization_plan = [
            (MOD_RGB, batch[MOD_RGB], rgb_tags),
            (MOD_DEPTH, batch[MOD_DEPTH], 'GT Depth'),
            (MOD_DEPTH, y_hat_depth, 'Prediction Depth'),
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
            y_hat_depth = y_hat[MOD_DEPTH]
            if type(y_hat_depth) is list:
                y_hat_depth = y_hat_depth[-1]
        self.visualize(batch,  y_hat_depth, rgb_tags, 'imgs_val/observed_samples')
        self.visualize_histograms(batch, y_hat_depth)

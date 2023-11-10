import torch
import torch.nn.functional as F

from source.models.model_parts import Encoder, get_encoder_channel_counts, ASPP, DecoderDeeplabV3p


class ModelDeepLabV3PlusMultiTask(torch.nn.Module):
    def __init__(self, cfg, outputs_desc):
        super().__init__()
        self.outputs_desc = outputs_desc
        semseg_ch = outputs_desc['semseg']
        depth_ch = outputs_desc['depth']

        self.encoder = Encoder(
            cfg.model_encoder_name,
            pretrained=cfg.pretrained,
            zero_init_residual=True,
            replace_stride_with_dilation=(False, False, True),
        )

        ch_out_encoder_bottleneck, ch_out_encoder_4x = get_encoder_channel_counts(cfg.model_encoder_name)

        aspps = {}
        decoders = {}

        # TODO: Implement aspps and decoders for each task as a ModuleDict.
        self.aspp_semseg = ASPP(ch_out_encoder_bottleneck, 256)
        self.aspp_depth = ASPP(ch_out_encoder_bottleneck, 256)

        self.decoder_semseg = DecoderDeeplabV3p(256, ch_out_encoder_4x, semseg_ch)
        self.decoder_depth = DecoderDeeplabV3p(256, ch_out_encoder_4x, depth_ch)

    def forward(self, x):
        input_resolution = (x.shape[2], x.shape[3])

        features = self.encoder(x)

        lowest_scale = max(features.keys())

        features_lowest = features[lowest_scale]

        features_tasks_semseg = self.aspp_semseg(features_lowest)
        features_tasks_depth = self.aspp_depth(features_lowest)

        predictions_4x_semseg, _ = self.decoder_semseg(features_tasks_semseg, features[4])
        predictions_4x_depth, _ = self.decoder_depth(features_tasks_depth, features[4])

        predictions_1x_semseg = F.interpolate(predictions_4x_semseg, size=input_resolution, mode='bilinear', align_corners=False)
        predictions_1x_depth = F.interpolate(predictions_4x_depth, size=input_resolution, mode='bilinear', align_corners=False)

        out = {}
        # TODO: implement the forward pass for each task as in the previous exercise.
        # However, now task's aspp and decoder need to be forwarded separately.

        for task, num_ch in self.outputs_desc.items():
            if task == "semseg":
                current_prediction = predictions_1x_semseg
                out[task] = current_prediction
            elif task == "depth":
                current_prediction = predictions_1x_depth.exp().clamp(0.1, 300.0)
                # be sure that depth is > 0, you can use other operators than exp
                out[task] = current_prediction
        
        return out

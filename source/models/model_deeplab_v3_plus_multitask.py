import torch
import torch.nn.functional as F

from source.models.model_parts import Encoder, get_encoder_channel_counts, ASPP, DecoderDeeplabV3p


class ModelDeepLabV3PlusMultiTask(torch.nn.Module):
    def __init__(self, cfg, outputs_desc):
        super().__init__()
        self.outputs_desc = outputs_desc

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

        for task, num_ch in self.outputs_desc.items():
            aspps[task] = ASPP(ch_out_encoder_bottleneck, 256)
            decoders[task] = DecoderDeeplabV3p(256, ch_out_encoder_4x, num_ch)
            
        self.aspps = torch.nn.ModuleDict(aspps)
        self.decoders = torch.nn.ModuleDict(decoders)

    def forward(self, x):
        input_resolution = (x.shape[2], x.shape[3])

        features = self.encoder(x)

        lowest_scale = max(features.keys())

        features_lowest = features[lowest_scale]

        out = {}

        # TODO: implement the forward pass for each task as in the previous exercise.
        # However, now task's aspp and decoder need to be forwarded separately.
        
        for task, _ in self.outputs_desc.items():
            features_tasks = self.aspps[task](features_lowest)
            predictions_4x, _ = self.decoders[task](features_tasks, features[4])
            predictions_1x = F.interpolate(predictions_4x, size=input_resolution, mode='bilinear', align_corners=False)
            out[task] = predictions_1x.exp().clamp(0.1, 300.0) if task == "depth" else predictions_1x
        
        return out

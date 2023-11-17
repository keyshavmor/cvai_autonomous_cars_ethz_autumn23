import torch
import torch.nn as nn
import torch.nn.functional as F

from source.models.model_parts import Encoder, get_encoder_channel_counts, DecoderDeeplabV3p, TransformerBlock, LatentsExtractor


class ModelAdaptiveDepth(torch.nn.Module):
    def __init__(self, cfg, outputs_desc):
        super().__init__()
        self.outputs_desc = outputs_desc
        ch_out = sum(outputs_desc.values())

        self.encoder = Encoder(
            cfg.model_encoder_name,
            pretrained=cfg.pretrained,
            zero_init_residual=True,
            replace_stride_with_dilation=(False, False, True),
        )

        ch_out_encoder_bottleneck, ch_out_encoder_4x = get_encoder_channel_counts(cfg.model_encoder_name)

        self.decoder = DecoderDeeplabV3p(ch_out_encoder_bottleneck, ch_out_encoder_4x, 256)

        self.conv3x3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

        # self.latents = nn.Parameter(torch.randn(1, cfg.num_bins, 256), requires_grad=True)
        # depth_values = torch.linspace(0.1, 100.0, cfg.num_bins).view(1, cfg.num_bins, 1, 1)
        # self.depth_values = nn.Parameter(depth_values, requires_grad=False)
        
        self.extractor = LatentsExtractor(cfg.num_bins)
        self.transformer_blocks = nn.Sequential(*[TransformerBlock(256, num_heads=cfg.num_heads, expansion=cfg.expansion) for _ in range(cfg.num_transformer_layers)])
        self.regressor = nn.Sequential(nn.Linear(256, 256),
                                       nn.LeakyReLU(),
                                       nn.Linear(256, 256),
                                       nn.LeakyReLU(),
                                       nn.Linear(256, 1),
                                       nn.ReLU())

    def forward(self, x):
        B, _, H, W = x.shape
        input_resolution = (H, W)

        features = self.encoder(x)

        lowest_scale = max(features.keys())

        features_lowest = features[lowest_scale]

        features_4x, _ = self.decoder(features_lowest, features[4]) # output channels = C = 256

        queries = self.conv3x3(features_4x) # [B, C, h, w] (C = 256)

        # TODO: Implement the adaptive extraction of bins and depth computation
        # 1. extract "latents" (i.e., learnable bins) with LatentsExtractor
        # 2. process them with a few layers of TrasformerBlocks
        # 3. for each latent, project it to a scalar value to obtain the depth
        #    value each corresponds to (which is learnable based on the image)
        # 4. for each pixel, compute the probabilities that it belongs to one of 
        #    the bins: softmax of the dot product between queries and latents
        # The predefined depth_values and latents, will become learnable and 
        # determined by the specific input
        
        latents = self.extractor(features_4x) # [B, C, h, w] -> [B, num_bins, C] (C=256)
        latents = self.transformer_blocks(latents) # [B, num_bins, C]
        
        bins = self.regressor(latents) # [B, num_bins, 1]
        bins = bins + 0.1
        bins = bins / bins.sum(dim=1, keepdim=True)  # [B, num_bins, 1]
        bins = F.pad(bins, (0, 0, 1, 0))  # [B, num_bins + 1, 1]
        bin_edges = torch.cumsum(bins, dim=1)  # [B, num_bins + 1, 1]
        bin_edges = 0.1 + (100 - 0.1) * bin_edges  # [B, num_bins + 1, 1]
        depth_values = (bin_edges[:, :-1, :] + bin_edges[:, 1:, :])*0.5  # [B, num_bins, 1]
        
        similarity = torch.einsum("bcij,bnc->bnij", queries, latents) #= [B, num_bins, h, w] = [B, C, h, w]. [B, num_bins, C]
        
        predictions_4x = (F.softmax(similarity, dim=1) * depth_values.reshape(B, -1, 1, 1)).sum(dim=1, keepdim=True)
        predictions_1x = F.interpolate(predictions_4x, size=input_resolution, mode='bilinear', align_corners=False).reshape(B, H, W)

        out = {}
        offset = 0

        for task, num_ch in self.outputs_desc.items():
            if task == "depth":
                out[task] = predictions_1x
            else:
                NotImplementedError
            offset += num_ch

        return out

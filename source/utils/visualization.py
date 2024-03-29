import matplotlib
import matplotlib.cm
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib.font_manager import findfont, FontProperties
from PIL import Image, ImageFont, ImageDraw
from torchvision.utils import make_grid

from source.datasets.definitions import MOD_RGB, MOD_SEMSEG, MOD_DEPTH


class ImageTextRenderer:
    def __init__(self, size=60):
        font_path = findfont(FontProperties(family='monospace'))
        self.font = ImageFont.truetype(font_path, size=size, index=0)
        self.size = size

    def print_gray(self, img_np_f, text, offs_xy, white=1.0):
        assert len(img_np_f.shape) == 2, "Image must be single channel"
        img_pil = Image.fromarray(img_np_f, mode='F')
        ctx = ImageDraw.Draw(img_pil)
        step = self.size // 15
        for dx, dy in ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)):
            ctx.text((offs_xy[0] + step * dx, offs_xy[1] + step * dy), text, font=self.font, fill=0.0)
        ctx.text(offs_xy, text, font=self.font, fill=white)
        return np.array(img_pil)

    def print(self, img_np_f, text, offs_xy, **kwargs):
        if len(img_np_f.shape) == 3:
            for ch in range(3):
                img_np_f[ch] = self.print_gray(img_np_f[ch], text, offs_xy, **kwargs)
        else:
            img_np_f = self.print_gray(img_np_f, text, offs_xy, **kwargs)
        return img_np_f


_text_renderers = dict()


def get_text_renderer(size):
    if size not in _text_renderers:
        _text_renderers[size] = ImageTextRenderer(size)
    return _text_renderers[size]


def img_print(*args, **kwargs):
    size = kwargs['size']
    del kwargs['size']
    renderer = get_text_renderer(size)
    return renderer.print(*args, **kwargs)


def tensor_print(img, caption, **kwargs):
    if isinstance(caption, str) and len(caption.strip()) == 0:
        return img
    assert img.dim() == 4 and img.shape[1] in (1, 3), 'Expecting 4D tensor with RGB or grayscale'
    offset = min(img.shape[2], img.shape[3]) // 100
    img = img.cpu()
    offset = (offset, offset)
    size = min(img.shape[2], img.shape[3]) // 15
    for i in range(img.shape[0]):
        tag = (caption if isinstance(caption, str) else caption[i]).strip()
        if len(tag) == 0:
            continue
        img_np = img_print(img[i].numpy(), tag, offset, size=size, **kwargs)
        img[i] = torch.from_numpy(img_np)
    return img


def prepare_rgb(img, rgb_mean, rgb_stddev):
    assert img.dim() == 4 and img.shape[1] == 3, 'Expecting 4D tensor with RGB'
    img = img.float().cpu()
    img_mean, img_stddev = torch.tensor(rgb_mean).float().view(3, 1, 1), torch.tensor(rgb_stddev).float().view(3, 1, 1)
    return torch.clamp((img * img_stddev + img_mean) / 255, 0, 1)


def superimpose_rgb(img, rgb, opacity=0.5):
    gray = rgb2gray(rgb)
    img = (1 - opacity) * gray + opacity * img
    return img


def rgb2gray(rgb):
    return rgb[:, 1, :, :].unsqueeze(1).repeat(1, 3, 1, 1)  # RGB -> GGG


def create_checkerboard(N, C, H, W):
    cell_sz = max(min(H, W) // 32, 1)
    mH = (H + cell_sz - 1) // cell_sz
    mW = (W + cell_sz - 1) // cell_sz
    checkerboard = torch.full((mH, mW), 0.25, dtype=torch.float32)
    checkerboard[0::2, 0::2] = 0.75
    checkerboard[1::2, 1::2] = 0.75
    checkerboard = checkerboard.float().view(1, 1, mH, mW)
    checkerboard = F.interpolate(checkerboard, scale_factor=cell_sz, mode='nearest')
    checkerboard = checkerboard[:, :, :H, :W].repeat(N, C, 1, 1)
    return checkerboard


def prepare_semseg(img, semseg_color_map, semseg_ignore_label):
    assert (img.dim() == 3 or img.dim() == 4 and img.shape[1] == 1) and img.dtype in (torch.int, torch.long), \
        f'Expecting 4D tensor with semseg classes, got {img.shape}'
    if img.dim() == 4:
        img = img.squeeze(1)
    colors = torch.tensor(semseg_color_map, dtype=torch.float32)
    assert colors.dim() == 2 and colors.shape[1] == 3
    if torch.max(colors) > 128:
        colors /= 255
    img = img.cpu().clone()  # N x H x W
    N, H, W = img.shape
    img_color_ids = torch.unique(img)
    assert all(c_id == semseg_ignore_label or 0 <= c_id < len(semseg_color_map) for c_id in img_color_ids)
    checkerboard, mask_ignore = None, None
    if semseg_ignore_label in img_color_ids:
        checkerboard = create_checkerboard(N, 3, H, W)
        mask_ignore = img == semseg_ignore_label
        img[mask_ignore] = 0
    img = colors[img]  # N x H x W x 3
    img = img.permute(0, 3, 1, 2)
    if semseg_ignore_label in img_color_ids:
        mask_ignore = mask_ignore.unsqueeze(1).repeat(1, 3, 1, 1)
        img[mask_ignore] = checkerboard[mask_ignore]
    return img


def prepare_mask(mask):
    assert mask.dim() == 4 and mask.shape[1] == 1
    mask = mask.float().cpu()
    out = torch.cat((mask, mask, mask), dim=1)
    return out


def collect_depth_range(img):
    mask_img_valid = img[img == img]
    if mask_img_valid.numel() == 0:
        return 0, 1
    valid_min = torch.min(mask_img_valid).cpu()
    valid_max = torch.max(mask_img_valid).cpu()
    return valid_min, valid_max


def prepare_depth(img, depth_min, depth_max, cmap=None):
    assert img.dim() == 3 or img.dim() == 4 and img.shape[1] == 1, "Invalid number of channels"
    if img.dim() == 4:
        img = torch.squeeze(img, dim=1)
    img_safe = img.clone()
    mask_img_invalid = img_safe != img_safe
    img_safe[mask_img_invalid] = 0
    cm = matplotlib.cm.get_cmap(cmap if cmap is not None else 'plasma_r')

    # scale depth range to [0,1]
    img_01 = ((img_safe - depth_min) / (depth_max - depth_min)).clamp(0, 1)
    # scale logarithmically to highlight close objects
    img_01 = (1 + img_01 * 1000).log() / np.math.log(1001)

    img_colored_np = cm(img_01.cpu().numpy(), bytes=False)[:,:,:,0:3]
    img_colored_np = np.rollaxis(img_colored_np, 3, 1)
    img_colored = torch.tensor(img_colored_np).float()
    mask_img_invalid = torch.unsqueeze(mask_img_invalid, dim=1)
    mask_img_invalid = torch.cat((mask_img_invalid, mask_img_invalid, mask_img_invalid), dim=1)
    checkerboard = create_checkerboard(*mask_img_invalid.shape)
    img_colored[mask_img_invalid] = checkerboard[mask_img_invalid]
    return img_colored


def compose(
        list_triples, cfg,
        rgb_mean=None, rgb_stddev=None, semseg_color_map=None, semseg_ignore_label=None, depth_range=None
):
    if cfg.visualize_num_samples_in_batch is not None:
        list_triples = [(m, img[:cfg.visualize_num_samples_in_batch], c) for (m, img, c) in list_triples]
    N, H, W, rgb = None, None, None, None
    stats_collected = []
    for i, (modality, img, caption) in enumerate(list_triples):
        if modality == MOD_RGB and MOD_RGB not in stats_collected:
            N, _, H, W = img.shape
            rgb = prepare_rgb(img, rgb_mean, rgb_stddev)
            stats_collected.append(MOD_RGB)
        if modality == MOD_DEPTH and MOD_DEPTH not in stats_collected:
            depth_range = collect_depth_range(img)
            stats_collected.append(MOD_DEPTH)
    vis = []
    for i, (modality, img, caption) in enumerate(list_triples):
        if modality == MOD_RGB:
            vis.append(tensor_print(
                rgb.clone(), caption if type(caption) is list else [str(idx) for idx in caption.tolist()])
            )
        elif modality == MOD_SEMSEG:
            vis.append(tensor_print(prepare_semseg(img, semseg_color_map, semseg_ignore_label), caption))
        elif modality == MOD_DEPTH:
            vis.append(tensor_print(prepare_depth(img, *depth_range), caption))
        else:
            assert False, f'Unknown modality {modality}'
    vis = torch.cat(vis, dim=2)
    # N x 3 x H * N_MODS x W
    vis = make_grid(vis, nrow=min(N, cfg.visualize_img_grid_width))
    return vis

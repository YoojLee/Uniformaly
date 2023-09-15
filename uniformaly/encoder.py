import torch
import timm
from torch.hub import HASH_REGEX, download_url_to_file, urlparse
from backbones import vision_transformer

import logging
import os

_logger = logging.getLogger(__name__)

_WEIGHTS_DIR = "backbones/weights"
os.makedirs(_WEIGHTS_DIR, exist_ok = True)

_BACKBONES = {
    "vit_small": 'timm.create_model("vit_small_patch16_224", pretrained=True)',
    "vit_large": 'timm.create_model("vit_large_patch16_224", pretrained=True)',
    "vit_r50": 'timm.create_model("vit_large_r50_s32_224", pretrained=True)',
    "vit_deit_base": 'timm.create_model("deit_base_patch16_224", pretrained=True)',
    "vit_deit_distilled": 'timm.create_model("deit_base_distilled_patch16_224", pretrained=True)',
    "vit_swin_base": 'timm.create_model("swin_base_patch4_window7_224", pretrained=True)',
    "vit_swin_large": 'timm.create_model("swin_large_patch4_window7_224", pretrained=True)',
}

def load(name):

    if name in _BACKBONES.keys():
        return eval(_BACKBONES[name])

    arch, patchsize = name.split("_")[-2], name.split("_")[-1]
    model = vision_transformer.__dict__[f'vit_{arch}'](patch_size=int(patchsize))

    if "dino" in name:
        if arch == "base":
            ckpt_pth = download_cached_file(f"https://dl.fbaipublicfiles.com/dino/dino_vit{arch}{patchsize}_pretrain/dino_vit{arch}{patchsize}_pretrain.pth")
        elif arch == "small":
            ckpt_pth = download_cached_file(f"https://dl.fbaipublicfiles.com/dino/dino_deit{arch}{patchsize}_pretrain/dino_deit{arch}{patchsize}_pretrain.pth")
        else:
            raise ValueError("Invalid type of architecture. It must be either 'small' or 'base'.")

        state_dict = torch.load(ckpt_pth, map_location='cpu')

    if "moco" in name:
        state_dict = convert_key(download_cached_file(f"https://dl.fbaipublicfiles.com/moco-v3/vit-{arch[0]}-300ep/vit-{arch[0]}-300ep.pth.tar"))

    if "mae" in name:
        ckpt_pth = download_cached_file(f"https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_{arch}.pth")
        state_dict = torch.load(ckpt_pth, map_location='cpu')['model']

    if "ibot" in name:
        if arch == 'small':
            ckpt_pth = "/home/workspace/backbones/weights/ibot_vit_small_16.pth"
        else:
            ckpt_pth = download_cached_file(f"https://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/archive/2022/ibot/vit{arch[0]}_{patchsize}_rand_mask/checkpoint_teacher.pth")
        state_dict = torch.load(ckpt_pth, map_location='cpu')['state_dict']

    if "beit" in name:
        ckpt_pth = download_cached_file(f"https://conversationhub.blob.core.windows.net/beit-share-public/beit/beit_{arch}_patch{patchsize}_224_pt22k.pth")
        state_dict = torch.load(ckpt_pth, map_location='cpu')['model']

    elif "sup" in name:
        try:
            state_dict = torch.load(f"{_WEIGHTS_DIR}/vit_{arch}_patch{patchsize}_in1k.pth")
        except FileNotFoundError:
            state_dict = torch.load(f"{_WEIGHTS_DIR}/vit_{arch}_patchsize_{patchsize}_224.pth")
            

    model.load_state_dict(state_dict, strict=False)
    return model


def download_cached_file(url, check_hash=True, progress=True):
        """
        Mostly copy-paste from timm library.
        (https://github.com/rwightman/pytorch-image-models/blob/29fda20e6d428bf636090ab207bbcf60617570ca/timm/models/_hub.py#L54)
        """
        if isinstance(url, (list, tuple)):
            url, filename = url
        else:
            parts = urlparse(url)
            filename = os.path.basename(parts.path)
        cached_file = os.path.join(_WEIGHTS_DIR, filename)
        if not os.path.exists(cached_file):
            _logger.info('Downloading: "{}" to {}\n'.format(url, cached_file))
            hash_prefix = None
            if check_hash:
                r = HASH_REGEX.search(filename)  # r is Optional[Match[str]]
                hash_prefix = r.group(1) if r else None
            download_url_to_file(url, cached_file, hash_prefix, progress=progress)
        return cached_file


def convert_key(ckpt_pth):
    ckpt = torch.load(ckpt_pth, map_location="cpu")
    state_dict = ckpt['state_dict']
    new_state_dict = dict()

    for k, v in state_dict.items():
        if k.startswith('module.base_encoder.'):
            new_state_dict[k[len("module.base_encoder."):]] = v

    return new_state_dict
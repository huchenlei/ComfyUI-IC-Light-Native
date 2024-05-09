""" Convert diffusers weight to ldm weight. """

import os
import folder_paths
import safetensors.torch

from comfy.diffusers_convert import convert_unet_state_dict


def convert_weight():
    src = "iclight_sd15_fbc.safetensors"
    dest = "iclight_sd15_fbc_unet_ldm.safetensors"

    ic_light_root = os.path.join(folder_paths.models_dir, "ic_light")
    model_path = os.path.join(ic_light_root, src)

    sd_dict = convert_unet_state_dict(safetensors.torch.load_file(model_path))
    sd_dict = {key: sd_dict[key].half() for key in sd_dict.keys()}
    safetensors.torch.save_file(sd_dict, dest)

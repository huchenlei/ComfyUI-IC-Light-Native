import os
import torch
from typing import Tuple, TypedDict, Callable

import folder_paths
import comfy.model_management
from comfy.diffusers_convert import convert_unet_state_dict
from comfy.model_patcher import ModelPatcher
from comfy.sd import VAE
from comfy.ldm.models.autoencoder import AutoencoderKL
from comfy.latent_formats import SD15 as SD15Config
from nodes import VAEEncode


if "ic_light" in folder_paths.folder_names_and_paths:
    ic_light_root = folder_paths.get_folder_paths("ic_light")[0]
else:
    ic_light_root = os.path.join(folder_paths.models_dir, "ic_light")


class UnetParams(TypedDict):
    input: torch.Tensor
    timestep: torch.Tensor
    c: dict
    cond_or_uncond: torch.Tensor


class VAEEncodeArgMax(VAEEncode):
    """Setting regularizer.sample = False to obtain mode of distribution."""

    def encode(self, vae: VAE, pixels):
        """@Override"""
        assert isinstance(
            vae.first_stage_model, AutoencoderKL
        ), "ArgMax only supported for AutoencoderKL"
        original_sample_mode = vae.first_stage_model.regularization.sample
        vae.first_stage_model.regularization.sample = False
        ret = super().encode(vae, pixels)
        vae.first_stage_model.regularization.sample = original_sample_mode
        return ret


class ICLight:
    """ICLightImpl"""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "c_concat": ("LATENT",),
            },
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply"
    CATEGORY = "_for_testing"

    def __init__(self) -> None:
        self.new_conv_in = None

    def apply(
        self,
        model: ModelPatcher,
        c_concat: dict,
    ) -> Tuple[ModelPatcher]:
        """ """
        device = comfy.model_management.get_torch_device()
        dtype = comfy.model_management.unet_dtype()
        work_model = model.clone()

        # Apply scale factor.
        scale_factor = SD15Config().scale_factor
        c_concat_samples: torch.Tensor = c_concat["samples"] * scale_factor

        def wrapped_unet(unet_apply: Callable, params: UnetParams):
            # Apply concat.
            sample = params["input"]
            params["c"]["c_concat"] = torch.cat(
                (
                    [c_concat_samples.to(sample.device)]
                    * (sample.shape[0] // c_concat_samples.shape[0])
                ),
                dim=0,
            )
            return unet_apply(x=sample, t=params["timestep"], **params["c"])

        work_model.set_model_unet_function_wrapper(wrapped_unet)
        # model_path = os.path.join(ic_light_root, "iclight_sd15_fc.safetensors")
        # model_path = os.path.join(ic_light_root, "merged_ic_light.safetensors")
        # sd_dict = convert_unet_state_dict(safetensors.torch.load_file(model_path))
        # sd_dict = {
        #     ("model.diffusion_model." + key): sd_dict[key]
        #     for key in sd_dict.keys()
        # }
        # safetensors.torch.save_file(sd_dict, "merged_ic_light.safetensors")

        # work_model.add_patches(
        #     patches={
        #         ("diffusion_model." + key): (
        #             sd_offset[key].to(dtype=dtype, device=device),
        #         )
        #         for key in sd_offset.keys()
        #     }
        # )
        return (work_model,)


NODE_CLASS_MAPPINGS = {
    "ICLightAppply": ICLight,
    "VAEEncodeArgMax": VAEEncodeArgMax,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ICLightApply": "IC Light Apply",
    "VAEEncodeArgMax": "VAEEncodeArgMax",
}

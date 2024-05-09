import os
import torch
import safetensors.torch
from typing import Tuple, TypedDict, Callable, NamedTuple

import folder_paths
import comfy.model_management
from comfy.diffusers_convert import convert_unet_state_dict
from comfy.model_patcher import ModelPatcher
from comfy.ldm.modules.diffusionmodules.openaimodel import UNetModel
from comfy.model_base import BaseModel
from comfy.conds import CONDRegular


# from comfy.ops import disable_weight_init as ops


if "ic_light" in folder_paths.folder_names_and_paths:
    ic_light_root = folder_paths.get_folder_paths("ic_light")[0]
else:
    ic_light_root = os.path.join(folder_paths.models_dir, "ic_light")


# @torch.inference_mode()
# def encode_prompt_inner(txt: str):
#     max_length = tokenizer.model_max_length
#     chunk_length = tokenizer.model_max_length - 2
#     id_start = tokenizer.bos_token_id
#     id_end = tokenizer.eos_token_id
#     id_pad = id_end

#     def pad(x, p, i):
#         return x[:i] if len(x) >= i else x + [p] * (i - len(x))

#     tokens = tokenizer(txt, truncation=False, add_special_tokens=False)["input_ids"]
#     chunks = [[id_start] + tokens[i: i + chunk_length] + [id_end] for i in range(0, len(tokens), chunk_length)]
#     chunks = [pad(ck, id_pad, max_length) for ck in chunks]

#     token_ids = torch.tensor(chunks).to(device=device, dtype=torch.int64)
#     conds = text_encoder(token_ids).last_hidden_state

#     return conds


# @torch.inference_mode()
# def encode_prompt_pair(positive_prompt, negative_prompt):
#     c = encode_prompt_inner(positive_prompt)
#     uc = encode_prompt_inner(negative_prompt)

#     c_len = float(len(c))
#     uc_len = float(len(uc))
#     max_count = max(c_len, uc_len)
#     c_repeat = int(math.ceil(max_count / c_len))
#     uc_repeat = int(math.ceil(max_count / uc_len))
#     max_chunk = max(len(c), len(uc))

#     c = torch.cat([c] * c_repeat, dim=0)[:max_chunk]
#     uc = torch.cat([uc] * uc_repeat, dim=0)[:max_chunk]

#     c = torch.cat([p[None, ...] for p in c], dim=1)
#     uc = torch.cat([p[None, ...] for p in uc], dim=1)

#     return c, uc


def apply_c_concat(cond, uncond, c_concat: torch.Tensor):
    """Set foreground/background concat condition."""

    def write_c_concat(cond):
        new_cond = []
        for t in cond:
            n = [t[0], t[1].copy()]
            if "model_conds" not in n[1]:
                n[1]["model_conds"] = {}
            n[1]["model_conds"]["c_concat"] = CONDRegular(c_concat)
            new_cond.append(n)
        return new_cond

    return (write_c_concat(cond), write_c_concat(uncond))


class UnetParams(TypedDict):
    input: torch.Tensor
    timestep: torch.Tensor
    c: dict
    cond_or_uncond: torch.Tensor


def create_custom_conv(
    original_conv: torch.nn.Module,
    dtype: torch.dtype,
    device=torch.device,
) -> torch.nn.Module:
    with torch.no_grad():
        new_conv_in = torch.nn.Conv2d(
            8,
            original_conv.out_channels,
            original_conv.kernel_size,
            original_conv.stride,
            original_conv.padding,
            dtype=dtype,
            device=device,
        )
        new_conv_in.weight.zero_()
        new_conv_in.weight[:, :4, :, :].copy_(original_conv.weight)
        new_conv_in.bias = original_conv.bias
        return new_conv_in


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
        base_model: BaseModel = work_model.model
        unet: UNetModel = base_model.diffusion_model
        c_concat_samples: torch.Tensor = c_concat["samples"]

        conv = unet.input_blocks[0][0]
        self.new_conv_in = create_custom_conv(conv, dtype=dtype, device=device)

        def wrapped_unet(unet_apply: Callable, params: UnetParams):
            # Apply concat.
            sample = params["input"]
            params["c"]["c_concat"] = torch.cat(
                (
                    [c_concat_samples.to(sample.device)]
                    * (sample.shape[0] // c_concat_samples.shape[0])
                )
                + params["c"].get("c_concat", []),
                dim=0,
            )

            unet.input_blocks[0][0] = self.new_conv_in

            try:
                return unet_apply(x=sample, t=params["timestep"], **params["c"])
            finally:
                # Restore in case the original model is used somewhere else.
                # TODO make patch replace of conv_in standard in ComfyUI.
                unet.input_blocks[0][0] = conv

        work_model.set_model_unet_function_wrapper(wrapped_unet)
        model_path = os.path.join(ic_light_root, "iclight_sd15_fc.safetensors")
        sd_offset = convert_unet_state_dict(safetensors.torch.load_file(model_path))

        work_model.add_patches(
            patches={
                ("diffusion_model." + key): (
                    sd_offset[key].to(dtype=dtype, device=device),
                )
                for key in sd_offset.keys()
            }
        )
        return (work_model,)


NODE_CLASS_MAPPINGS = {
    "ICLightAppply": ICLight,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ICLightApply": "IC Light Apply",
}

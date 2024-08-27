# ComfyUI-IC-Light-Native
ComfyUI native implementation of [IC-Light](https://github.com/lllyasviel/IC-Light).

## Install
Download the repository and unpack into the custom_nodes folder in the ComfyUI installation directory.

Or clone via GIT, starting from ComfyUI installation directory:
```bash
cd custom_nodes
git clone git@github.com:huchenlei/ComfyUI-IC-Light-Native.git
```

### Download models
IC-Light main repo is based on diffusers. In order to load it with UnetLoader in ComfyUI, state_dict keys need to convert to ldm format. You can download models with ldm keys here: https://huggingface.co/huchenlei/IC-Light-ldm/tree/main

There are 2 models:
- iclight_sd15_fc_unet_ldm: Use this in FG workflows
- iclight_sd15_fbc_unet_ldm: Use this in BG workflows

After you download these models, please put them under `ComfyUI/models/unet` and load them with `UNETLoader` node.

### Recommended nodes

- [ComfyUI-KJNodes](https://github.com/kijai/ComfyUI-KJNodes): Provides various mask nodes to create light map.
- [ComfyUI-Easy-Use](https://github.com/yolain/ComfyUI-Easy-Use): A giant node pack of everything. The remove bg node used in workflow comes from this pack.
- [ComfyUI_essentials](https://github.com/cubiq/ComfyUI_essentials): Many useful tooling nodes. Image resize node used in the workflow comes from this pack.
- [ComfyUI-IC-Light](https://github.com/kijai/ComfyUI-IC-Light): The IC-Light impl from kijai. It includes a very useful `DetailTransfer` node to help preverse high frequency details from input fg image.

## Workflows
Please make sure the fg image's masked/transparent area are grey before you pass it to the VAE. Otherwise, you will get background obscured in FC workflows or
darkened background in FBC workflows. You can use `IC Light Apply Mask Grey` to make sure the masked area's color is correct. See following examples:
![12 05 2024_16 22 48_REC](https://github.com/huchenlei/ComfyUI-IC-Light-Native/assets/20929282/702c7b3a-54f7-44e2-a6d7-39220aa6ffef)
![12 05 2024_16 19 02_REC](https://github.com/huchenlei/ComfyUI-IC-Light-Native/assets/20929282/6d2d504f-65be-47cb-9597-2e64fe239939)

### [Given FG, Generate BG and relight](https://github.com/huchenlei/ComfyUI-IC-Light/blob/main/examples/fg.json)
![image](https://github.com/huchenlei/ComfyUI-IC-Light-Native/assets/20929282/6b801a2d-f37c-44f4-b52d-ad7de1748f8e)
If you want to keep the original color of the fg object, you can put the fg object in the latent space to further guide the generation. [workflow](https://github.com/huchenlei/ComfyUI-IC-Light-Native/blob/main/examples/ic_light_preserve_color.json)
![image](https://github.com/huchenlei/ComfyUI-IC-Light-Native/assets/20929282/6fb8c01e-727c-4fa7-b72f-f91ea3dce004)

### [Given FG and light map, Genereate BG and relight](https://github.com/huchenlei/ComfyUI-IC-Light/blob/main/examples/fg_lightmap.json)
Light from right
![image](https://github.com/huchenlei/ComfyUI-IC-Light-Native/assets/20929282/045e4f0e-6083-496f-af32-41de4821afbf)

Light from left
![image](https://github.com/huchenlei/ComfyUI-IC-Light-Native/assets/20929282/74750b9e-bda7-43f7-944f-d75cb7b5fb7e)

### [Given FG and BG, Put FG on BG and relight](https://github.com/huchenlei/ComfyUI-IC-Light/blob/main/examples/fg_bg_combine.json)
![image](https://github.com/huchenlei/ComfyUI-IC-Light-Native/assets/20929282/ea87538a-15d8-43d8-874d-bcddab9f4f0e)

### [Recover high frequency detail (Text, etc) from original input image](https://github.com/huchenlei/ComfyUI-IC-Light-Native/blob/main/examples/ic_light_detail_transfer.json)
![06 06 2024_12 19 35_REC](https://github.com/huchenlei/ComfyUI-IC-Light-Native/assets/20929282/7fbd66e9-5468-4644-9edb-abbc5aa55b77)
- Input image:
  ![image](https://github.com/huchenlei/ComfyUI-IC-Light-Native/assets/20929282/ebdd9f49-41ee-47e3-a334-299fd1ee0385)
- Raw generation:
  ![image](https://github.com/huchenlei/ComfyUI-IC-Light-Native/assets/20929282/040a59ff-1aaf-4df9-bb00-1e50ae67db1e)
- After detail transfer:
  ![image](https://github.com/huchenlei/ComfyUI-IC-Light-Native/assets/20929282/86ddfc4d-7077-43ea-979f-8dcf243aaaf9)

## Common Issues
IC-Light's unet is accepting extra inputs on top of the common noise input. FG model accepts extra 1 input (4 channels). BG model accepts 2 extra input (8 channels).
The original unet's input is 4 channels as well.

If you see following error, it means you are using FG workflow but loaded the BG model.
```
RuntimeError: Given groups=1, weight of size [320, 8, 3, 3], expected input[2, 12, 64, 64] to have 8 channels, but got 12 channels instead
```

If you see following error, it means you are using FG workflow but loaded the BG model.
```
RuntimeError: Given groups=1, weight of size [320, 12, 3, 3], expected input[2, 8, 64, 64] to have 12 channels, but got 8 channels instead
```

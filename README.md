# ComfyUI-IC-Light
ComfyUI native implementation of [IC-Light](https://github.com/lllyasviel/IC-Light).

## Install
Download the repository and unpack into the custom_nodes folder in the ComfyUI installation directory.

Or clone via GIT, starting from ComfyUI installation directory:
```bash
cd custom_nodes
git clone git@github.com:huchenlei/ComfyUI-IC-Light.git
```

### Download models
IC-Light main repo is based on diffusers. In order to load it with UnetLoader in ComfyUI, state_dict keys need to convert to ldm format. You can download models with ldm keys here: https://huggingface.co/huchenlei/IC-Light-ldm/tree/main

There are 2 models:
- iclight_sd15_fc_unet_ldm: Use this in FG workflows
- iclight_sd15_fbc_unet_ldm: Use this in BG workflows

After you download these models, please put them under `ComfyUI/models/unet` and load them with `UNETLoader` node.

### [Important!] Required nodes
You MUST install following nodes first for IC light to work properly.
- [ComfyUI-layerdiffuse](https://github.com/huchenlei/ComfyUI-layerdiffuse): Although not used in the workflow, the patching of weight load in layerdiffuse is a dependency for IC-Light nodes to work properly.
  
### Recommended nodes
- [ComfyUI-KJNodes](https://github.com/kijai/ComfyUI-KJNodes): Provides various mask nodes to create light map.
- [ComfyUI-Easy-Use](https://github.com/yolain/ComfyUI-Easy-Use): A giant node pack of everything. The remove bg node used in workflow comes from this pack.
- [ComfyUI_essentials](https://github.com/cubiq/ComfyUI_essentials): Many useful tooling nodes. Image resize node used in the workflow comes from this pack.

## Workflows
### [Given FG, Generate BG and relight](https://github.com/huchenlei/ComfyUI-IC-Light/blob/main/examples/fg.json)
![image](https://github.com/huchenlei/ComfyUI-IC-Light/assets/20929282/cde6da71-c944-4982-9e2d-eab57c85c3cf)

### [Given FG and light map, Genereate BG and relight](https://github.com/huchenlei/ComfyUI-IC-Light/blob/main/examples/fg_lightmap.json)
Light from right
![image](https://github.com/huchenlei/ComfyUI-IC-Light/assets/20929282/6b290e52-59cc-4d0a-8d48-17750cba5cc6)
Light from left
![image](https://github.com/huchenlei/ComfyUI-IC-Light/assets/20929282/6962c9b6-dc12-4876-8cd1-75428f6549f1)

### [Given FG and BG, Put FG on BG and relight](https://github.com/huchenlei/ComfyUI-IC-Light/blob/main/examples/fg_bg_combine.json)
![image](https://github.com/huchenlei/ComfyUI-IC-Light/assets/20929282/30c5c210-2636-4f8f-9719-738fa0e377ca)

## Common Issues
IC-Light's unet is accepting extra inputs on top of the common noise input. FG model accepts extra 1 input (4 channels). BG model accepts 2 extra input (8 channels).
The original unet's input is 4 channels as well. 

If you see following issue, it means IC-Light's unet is not properly loaded, and you need to install [ComfyUI-layerdiffuse](https://github.com/huchenlei/ComfyUI-layerdiffuse) first.
```
RuntimeError: Given groups=1, weight of size [320, 4, 3, 3], expected input[2, 8, 64, 64] to have 4 channels, but got 8 channels instead
```

If you see following error, it means you are using FG workflow but loaded the BG model.
```
RuntimeError: Given groups=1, weight of size [320, 8, 3, 3], expected input[2, 12, 64, 64] to have 8 channels, but got 12 channels instead
```

If you see following error, it means you are using FG workflow but loaded the BG model.
```
RuntimeError: Given groups=1, weight of size [320, 12, 3, 3], expected input[2, 8, 64, 64] to have 12 channels, but got 8 channels instead
```

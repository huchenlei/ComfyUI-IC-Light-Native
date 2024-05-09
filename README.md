# ComfyUI-IC-Light
ComfyUI native implementation of [IC-Light](https://github.com/lllyasviel/IC-Light).

## Install
### Download models
IC-Light main repo is based on diffusers. In order to load it with UnetLoader in ComfyUI, state_dict keys need to convert to ldm format. You can download models with ldm keys here: https://huggingface.co/huchenlei/IC-Light-ldm/tree/main
There are 2 models:
- iclight_sd15_fc_unet_ldm: Use this in FG workflows
- iclight_sd15_fbc_unet_ldm: Use this in BG workflows
After you download these models, please put them under `ComfyUI/models/unet` and load them with `UNETLoader` node.

### Required nodes
- [ComfyUI-layerdiffuse](https://github.com/huchenlei/ComfyUI-layerdiffuse): Although not used in the workflow, the patching of weight load in layerdiffuse is a dependency for IC-Light nodes to work properly.
  
### Recommended nodes
- [ComfyUI-KJNodes](https://github.com/kijai/ComfyUI-KJNodes): Provides various mask nodes to create light map.
- [ComfyUI-Easy-Use](https://github.com/yolain/ComfyUI-Easy-Use): A giant node pack of everything. The remove bg node used in workflow comes from this pack.
- [ComfyUI_essentials](https://github.com/cubiq/ComfyUI_essentials): Many useful tooling nodes. Image resize node used in the workflow comes from this pack.

## Workflows
### Given FG, Generate BG and relight
![image](https://github.com/huchenlei/ComfyUI-IC-Light/assets/20929282/b3dd0332-685f-41d6-aa4e-3ebfce480df7)

### Given FG and light map, Genereate BG and relight
Light from right
![image](https://github.com/huchenlei/ComfyUI-IC-Light/assets/20929282/4677eda3-5f2a-4948-8051-2fb7fc94f734)
Light from left
![image](https://github.com/huchenlei/ComfyUI-IC-Light/assets/20929282/ad24d316-1237-4fb6-8e23-aeef88a24bf7)

### Given FG and BG, Put FG on BG and relight
![image](https://github.com/huchenlei/ComfyUI-IC-Light/assets/20929282/30c5c210-2636-4f8f-9719-738fa0e377ca)

## TODO
- [ ] How to use/install guide
- [ ] Model download links

# segment-anything-2 real-time
Run Segment Anything Model 2 on a **live video stream**

## News
- 13/12/2024 : Update to sam2.1
- 20/08/2024 : Fix management of ```non_cond_frame_outputs``` for better performance and add bbox prompt

## Demos
<div align=center>
<p align="center">
<img src="./assets/blackswan.gif" width="880">
</p>

</div>



## Getting Started

### Installation

```bash
pip install -e .
```
### Download Checkpoint

Then, we need to download a model checkpoint.

```bash
cd checkpoints
./download_ckpts.sh
```

Then SAM-2-online can be used in a few lines as follows for image and video and **camera** prediction.

### Camera prediction

```python
import torch
from sam2.build_sam import build_sam2_camera_predictor

sam2_checkpoint = "../checkpoints/sam2.1_hiera_small.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_s.yaml"
predictor = build_sam2_camera_predictor(model_cfg, checkpoint)

cap = cv2.VideoCapture(<your video or camera >)

if_init = False

with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        width, height = frame.shape[:2][::-1]

        if not if_init:
            predictor.load_first_frame(frame)
            if_init = True
            _, out_obj_ids, out_mask_logits = predictor.add_new_prompt(<your promot >)

        else:
            out_obj_ids, out_mask_logits = predictor.track(frame)
            ...
```

### With model compilation

You can use the `vos_inference` argument in the `build_sam2_camera_predictor` function to enable model compilation. The inference may be slow for the first few execution as the model gets warmed up, but should result in significant inference speed improvement. 

We provide the modified config file `sam2/configs/sam2.1/sam2.1_hiera_t_512.yaml`, with the modifications necessary to run SAM2 at a 512x512 resolution. Notably the parameters that need to be changed are highlighted in the config file at lines 24, 43, 54 and 89.

We provide the file `sam2/benchmark.py` to test the speed gain from using the model compilation.

## References:

- SAM2 Repository: https://github.com/facebookresearch/sam2

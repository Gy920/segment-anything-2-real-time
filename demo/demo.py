import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2


# use bfloat16 for the entire notebook
torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

if torch.cuda.get_device_properties(0).major >= 8:
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

from sam2.build_sam import build_sam2_camera_predictor
import time


sam2_checkpoint = "../checkpoints/sam2_hiera_small.pt"
model_cfg = "sam2_hiera_s.yaml"

predictor = build_sam2_camera_predictor(model_cfg, sam2_checkpoint)


cap = cv2.VideoCapture("../assets/blackswan.mp4")

if_init = False

while True:
    ret, frame = cap.read()
    width, height = frame.shape[:2][::-1]

    if not ret:
        break

    if not if_init:
        predictor.load_first_frame(frame)
        if_init = True

        ann_frame_idx = 0  # the frame index we interact with
        ann_obj_id = 2  # give a unique id to each object we interact with (it can be any integers)
        # Let's add a positive click at (x, y) = (210, 350) to get started
        points = np.array([[376, 325]], dtype=np.float32)
        # for labels, `1` means positive click and `0` means negative click
        labels = np.array([1], np.int32)
        _, out_obj_ids, out_mask_logits = predictor.add_new_points(
            frame_idx=ann_frame_idx,
            obj_id=ann_obj_id,
            points=points,
            labels=labels,
        )

    else:
        out_obj_ids, out_mask_logits = predictor.track(frame)

        all_mask = np.zeros((height, width, 1), dtype=np.uint8)
        print(all_mask.shape)
        for i in range(0, len(out_obj_ids)):
            out_mask = (out_mask_logits[i] > 0.0).permute(1, 2, 0).cpu().numpy().astype(
                np.uint8
            ) * 255

            all_mask = cv2.bitwise_or(all_mask, out_mask)

        print(all_mask.shape, type(all_mask))
        frame = cv2.bitwise_and(frame, frame, mask=out_mask)

    cv2.imshow("frame", frame)
    if cv2.waitKey(0) & 0xFF == ord("q"):
        break

# Dense Pedestrian Detection Based on Deformable DETR

This project focuses on dense pedestrian detection based on Deformable DETR using the MMDetection framework, with adaptation to the CrowdHuman dataset.

## Description
Deformable DETR is adapted for single-class pedestrian detection by converting CrowdHuman annotations into the required format.  
To address dense and heavily occluded scenarios, multi-scale resizing and large-scale random cropping strategies are applied during training.

The model is initialized with COCO pre-trained weights and trained using Warmup and MultiStep learning rate scheduling.  
AMP and EMA are enabled to improve training stability and convergence.  
On the validation set, the model achieves an mAP of 0.433 and AP50 of 0.765, significantly improving recall in dense crowd scenes compared to the original COCO-pretrained baseline.

## Features
- Deformable DETR based dense pedestrian detection
- CrowdHuman dataset adaptation
- Multi-scale training and large-scale cropping
- AMP + EMA for stable convergence
- Improved recall in crowded scenes

## Demo
![demo](assets/demo_detr_crowdhuman.gif)

## Notes
- Dataset and trained weights are not included.
- This repository focuses on configuration design and training strategy adaptation using MMDetection.

## Acknowledgement
This project is based on Deformable DETR and MMDetection.

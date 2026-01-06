python tools/analysis_tools/compare_video.py \
  --config1 configs/deformable_detr/deformable-detr_r50_16xb2-50e_coco.py \
  --checkpoint1 /home/didu/checkpoints/deformable-detr_r50_16xb2-50e_coco_20221029_210934-6bc7d21b.pth \
  --config2 configs/deformable_detr/deformable-detr_r50_16xb2-50e_crowdhuman.py \
  --checkpoint2 /home/didu/work_dirs/deformable_detr_crowdhuman_r50/epoch_80.pth \
  --video /home/didu/demo_materials/human.mp4 \
  --out /home/didu/work_dirs/deformable_detr_crowdhuman_r50/20251128_135102/compare_pretrain_vs_crowdhuman.mp4 \
  --score-thr 0.5 \
  --label1 "COCO Pretrain" \
  --label2 "CrowdHuman Finetune"

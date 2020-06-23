python logits_fusion.py \
--test_json_path ./data/CIHP/crop.json \
--global_output_dir ./data/CIHP/global_pic_parsing \
--msrcnn_output_dir ./data/CIHP/crop_pic_parsing \
--gt_output_dir ./data/CIHP/crop_pic_parsing \
--mask_output_dir ./data/CIHP/crop_mask \
--save_dir ./data/CIHP/mhp_fusion_parsing
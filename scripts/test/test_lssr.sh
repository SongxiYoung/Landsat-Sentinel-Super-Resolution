
python test_lssr.py \
--pretrained_model_path stable-diffusion-2-1-base \
--pretrained_path experiments/deploy/checkpoints/model_3501.pkl \
--dataset_txt_paths="preset/gt_path_tif_test_v2.txt" \
--highquality_dataset_txt_paths="preset/gt_selected_path_tif_test_v2.txt" \
--dem_dataset_txt_paths="preset/dem_tif_test_v2.txt" \
--landcover_dataset_txt_paths="preset/landcover_tif_test_v2.txt" \
--s1_dataset_txt_paths="preset/s1_tif_test_v2.txt" \
--process_size 64 \
--resolution_tgt 192 \
--upscale 3 \
--output_dir experiments/deploy \
--lambda_pix 1.0 \
--lambda_sem 1.0
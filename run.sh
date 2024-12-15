#DIR=fsdv2
#WORK=work_dirs
#CONFIG=fsdv2_waymo_1x
#bash tools/dist_train.sh configs/$DIR/$CONFIG.py 8 --work-dir ./$WORK/$CONFIG/ --cfg-options evaluation.pklfile_prefix=./$WORK/$CONFIG/results evaluation.metric=fast --seed 1

# CTRL training
# DIR=ctrl
# CONFIG=ctrl_veh_24e
# bash tools/dist_train.sh configs/$DIR/$CONFIG.py 8 --work-dir ./$WORK/$CONFIG/ --no-validate

# fast Waymo Evaluation, for all waymo-based models
# bash tools/dist_test.sh configs/$DIR/$CONFIG.py ./$WORK/$CONFIG/latest.pth 8 --options "pklfile_prefix=./$WORK/$CONFIG/results" --eval fast

cd /220019043/codes/SST/
#/root/miniconda3/envs/mmdet/bin/python  ./tools/occ/occ_annotate.py --split training --bin-path ./data/waymo/waymo_format/train_gt.bin --workers 64 --out-dir /LiZhen_team/dataset/waymo/waymo_occ_gt/ --chunksize 1 --ngpus 6
/root/miniconda3/envs/mmdet/bin/python ./tools/occ/occ_annotate.py --bin-path data/waymo/waymo_format/gt.bin --workers 16 --chunksize 1 --voxel-size 0.1 --out-dir /LiZhen_team/dataset/waymo/waymo_occ_gt_01vs_val
#bash tools/dist_train.sh configs/occ/tracklet_occ_two_stage.py 8 --work-dir work_dirs/tracklet_occ_two_stage_tune_ae_full_train_repeat2_12e_occ_only/ --no-validate
#bash tools/dist_test.sh configs/occ/tracklet_occ_two_stage_iou_test.py work_dirs/tracklet_occ_two_stage_no_ae_pretrained/latest.pth 8 --eval waymo
#bash tools/dist_test.sh configs/occ/tracklet_occ_two_stage.py work_dirs/tracklet_occ_two_stage_tune_ae_gt/latest.pth 8 --eval waymo --options "pklfile_prefix=./work_dirs/tracklet_occ_two_stage_tune_ae_gt/result"
#/root/miniconda3/envs/mmdet/bin/python ./tools/ctrl/generate_candidates.py ./tools/ctrl/data_configs/gt_base_vehicle.yaml --process 8
# bash tools/dist_test.sh configs/occ/tracklet_occ_two_stage_iou_test.py  work_dirs/tracklet_occ_two_stage_tune_ae_full_train_repeat2_12e_occ_only/latest.pth  8 --eval iou
#bash tools/dist_train.sh configs/occ/fsd_waymowithoccD1_1x_1f_car_6e_tanh.py 8 --work-dir work_dirs/fsd_waymowithoccD1_1x_1f_car_6e_tanh_dbsample_nodispaste/

#/root/miniconda3/envs/mmdet/bin/python tools/create_data.py waymo --root-path ./data/waymo/ --out-dir ./data/waymo/ --workers 64 --extra-tag waymo
#bash tools/dist_test.sh configs/occ/tracklet_occ_two_stage.py work_dirs/tracklet_occ_two_stage_tune_ae_gt_full_trained_repeat2_12e/latest.pth 8 --options "pklfile_prefix=./work_dirs/tracklet_occ_two_stage_tune_ae_gt_full_trained_repeat2_12e/result_train"  --eval waymo
# bash tools/dist_train.sh configs/occ/fsd_waymowithoccD1_1x_6f_car.py 4
#bash tools/dist_train.sh configs/occ/tracklet_occ_two_stage.py 8 --work-dir work_dirs/tracklet_occ_two_stage_tune_ae_full_train_reg8_gttrack/ --no-validate
#bash tools/dist_train.sh configs/occ/fsd_waymowithoccD1_1x_1f_car_6e_tanh.py 8 --work-dir work_dirs/fsd_waymowithoccD1_1x_1f_car_12e_tanh_dbsample_minlength10_fsd1fbase_random05paste_64vfe/
#bash tools/dist_train.sh configs/ctrl/trans_veh_24e.py 4 --work-dir work_dirs/trans_veh_24e_fsd1fbase_train/ --no-validate
#bash tools/dist_train.sh configs/occ/tracklet_occ_two_stage_v4.py 8 --work-dir work_dirs/tracklet_occ_two_stage_v4_train_42e_fsd1fbase_no_empty/
#bash tools/dist_train.sh configs/occ/tracklet_occ_two_stage_v5.py 8 --work-dir work_dirs/tracklet_occ_two_stage_v5_fsd1fbase_rcnn_single_combinedata_24e_freeae --resume-from work_dirs/tracklet_occ_two_stage_v5_fsd1fbase_rcnn_single_combinedata_24e_freeae/latest.pth
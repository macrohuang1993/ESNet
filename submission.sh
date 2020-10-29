model_path=/mnt/wekanfs/scratch/zhengyu.huang/FADNet_result/models/kitti_finetune/pretrained_on_sceneflow_run1/best.tar
save_path=./submit_results/fadnet-KITTI2015-split_run1/
net=fadnet
# model_path=./trained/psmnet-imn-KITTI2015-split/best.tar
# save_path=./submit_results/psmnet-imn-KITTI2015-split/
# net=psmnet
python3 kitti_submission.py --maxdisp 192 \
                     --model $net \
                     --KITTI 2015 \
                     --datapath /mnt/wekanfs/scratch/zhengyu.huang/KITTI_2015_Stereo/testing/ \
                     --savepath $save_path \
                     --loadmodel $model_path \

# kitti 2015
#savemodel: which directory to save model
#loadmodel: where to load model to be fine tunned.
python3 kitti_finetune.py --maxdisp 192 \
                   --model esnet \
                   --devices 0,1 \
                   --datatype 2015 \
                   --datapath /mnt/wekanfs/scratch/zhengyu.huang/KITTI_2015_Stereo/training/ \
                   --loss loss_configs/kitti.json \
                   --savemodel  kitti_models/ \
                   --loadmodel none



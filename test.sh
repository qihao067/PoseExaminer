######## -- testing -- #######
python3 tools/test.py configs/pare/hrnet_w32_conv_pare_mix_no_mosh.py \
    --work-dir=./Methods/temp \
    ./Methods/PARE/ours/exp_best_pare.pth \
    --metrics pa-mpjpe mpjpe pve \
    --name 3dpw
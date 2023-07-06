####### -- training -- #######
num_gpu=8
num_proc=40

## clear all previous results
rm -rf exps
rm -rf evaluation_results.txt

## new experiments direct
ROOTNAME=ADV_TRAINING_PARE 
rm -rf $ROOTNAME
mkdir $ROOTNAME

# PARE
cp Methods/PARE/master/hrnet_w32_conv_pare.pth $ROOTNAME/adv_loop_0.pth 
CONFIG=configs/pare/hrnet_w32_conv_pare_mix_no_mosh_advTraining.py

for num in 0 1 2 3 4 5
do

    cur_num=`expr $num + 1`
    
    ######## Step 1: adv testing on current model 
    rm -rf a.log 
    rm -rf exps
    mkdir exps
    python3 tools/train_advTesting.py $CONFIG --num_proc ${num_proc} --num_gpu ${num_gpu} \
        --loop_num ${cur_num} --work-dir=./$ROOTNAME/advTesting_$num \
        --checkpoint=./$ROOTNAME/adv_loop_$num.pth --metrics pa-mpjpe mpjpe mpjpe-2d

    python3 tools/test_advTesting.py $CONFIG --num_proc ${num_proc} --num_gpu ${num_gpu} \
        --work-dir=./$ROOTNAME/advTesting_$num --checkpoint=./$ROOTNAME/adv_loop_$num.pth \
        --metrics pa-mpjpe mpjpe mpjpe-2d

    python3 tools/test_final.py ${num_proc} ./$ROOTNAME/adv_loop_$num.pth

    ######## Step 2: sample 40*500 failure cases for training
    python3 tools/train_advTraining.py --num_proc ${num_proc} --num_gpu ${num_gpu} \
        --loop_num ${cur_num} --root_name ${ROOTNAME}
    
    ######## Step 3:  train mmhuman
    DIRNAME=tools
    WORK_DIR=./$ROOTNAME/advTraining_$cur_num
    RESUME=./$ROOTNAME/adv_loop_$num.pth
    PORT=${PORT:-29500}
    PYTHONPATH="$DIRNAME/..":$PYTHONPATH \
    python3 -m torch.distributed.launch --nproc_per_node=${num_gpu} --master_port=$PORT \
        $DIRNAME/train.py $CONFIG --loop_num ${cur_num} --work-dir=${WORK_DIR} --launcher pytorch ${@:4} --no-validate \
        --resume-from $RESUME

    ######## step 5: save the results 
    cp ./$ROOTNAME/advTraining_$cur_num/latest.pth ./$ROOTNAME/adv_loop_$cur_num.pth

    ######## step 4: test on current test set and on 3dpw
    python3 tools/test.py $CONFIG --work-dir=./$ROOTNAME/advTesting_$cur_num ./$ROOTNAME/adv_loop_$cur_num.pth \
        --metrics pa-mpjpe mpjpe pve --name advTraining_loop_$cur_num-3dpw

    ######## step 6: save the results 
    mv exps ./$ROOTNAME/exps_loop_$cur_num # need to modify

done

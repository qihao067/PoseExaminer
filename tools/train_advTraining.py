import argparse
import copy
import os
import os.path as osp
import time
import sys
sys.path.append('./utils_PoseExaminer/')

import torch
#### for adv testing
import multiprocessing
import time
import numpy as np

from utils_PoseExaminer.render_pose import PoseRender


def parse_args():
    parser = argparse.ArgumentParser(description='Train a model')
    parser.add_argument('--num_proc', type=int, default=40)
    parser.add_argument('--loop_num', type=int, default=-1)
    parser.add_argument('--texture_id', type=int, default=-1)
    parser.add_argument('--root_name', type=str, default="exps")
    parser.add_argument('--num_gpu', type=int, default=8)
    args = parser.parse_args()

    return args

def load_results(proc_id,device):
    f = open('./exps/Proc_'+str(proc_id)+'_training/training_loss.txt','r')
    lines = f.readlines()
    line = lines[-1]
    line = line.split('[')
    id = int(line[1].split(']')[0])

    checkpoint_path = os.path.join('./exps/Proc_'+str(proc_id)+'_training','checkpoints')
    policy = np.load(os.path.join(checkpoint_path,'policy_%.5d.npz'%id))

    r_zgen = policy['Zgen']
    r_top = policy['noise_rand_top']
    r_bottom = policy['noise_rand_bottom']

    return torch.from_numpy(r_zgen).to(device), torch.from_numpy(r_top).to(device), torch.from_numpy(r_bottom).to(device)

def samples_and_save(proc_id, texture_id, num_gpu, sam_num, att_texture, att_occ, random_glo, random_bkgd):
    
    args=parse_args()

    time.sleep((proc_id-1)*2)

    num_gpu = args.num_gpu
    gpu_id = proc_id % num_gpu
    # gpu_list=['0','1','2','3','4','5','6','7']
    gpu_list = list(range(0, num_gpu))
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_list[gpu_id])
    ##############

    save_path = './exps/Proc_'+str(proc_id)+'/checkpoints'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Params
    if att_texture:
        raise NotImplementedError("You need to first give the list of UV map, then use this")
        import random
        texture_list = list(range(100)) 
        texture_id = random.choice(texture_list)
    else:
        texture_id = texture_id

    with_bkground = random_bkgd

    phase = 5 # sample for adv training
    
    ##########
    ########## initial smpl pose model
    num_poses = sam_num
    uv_pose_render = PoseRender(texture_id=texture_id,num_poses=num_poses,proc_id=proc_id,withOcc=att_occ)

    ##### Run Learning Algorithm #####
    if True:
        start_time = time.time()
        print("--- [TIME] run session: %s seconds ---" % (time.time() - start_time))

        current_time = time.time()

        with torch.no_grad():
            sample_dim = None
            p_action, high, low = load_results(proc_id,device)

            if True:
                if (high+low).mean()<0.08:
                    high = high+0.2
                    low = low+0.2
                
                print("--- [TIME] from policy to action : %s seconds ---" % (time.time() - current_time))
                current_time = time.time()
                
                # generate new test samples with current action
                uv_pose_render.render(p_action, low=low, high=high, with_bkground=with_bkground, phase=phase, sample_dim=sample_dim, random_glo=random_glo)

                print("--- [TIME] Rendering : %s seconds ---" % (time.time() - current_time))
                current_time = time.time()
            

def sample_poses(num_proc=1, texture_id=-1, num_gpu=8, sam_num=1, att_texture=False, att_occ=False, random_glo=False, random_bkgd=False):
    multiprocessing.set_start_method("forkserver")

    #
    for i in range(1,num_proc+1):
        # os.rename('./exps/Proc_'+str(i), './exps/Proc_'+str(i)+'_training')
        os.makedirs('./exps/Proc_'+str(i)+'')
        os.makedirs('./exps/Proc_'+str(i)+'/checkpoints')
        os.makedirs('./exps/Proc_'+str(i)+'/data_generated')
        os.makedirs('./exps/Proc_'+str(i)+'/visulization')
    
    if num_proc == 1:
        samples_and_save(1, texture_id, num_gpu, sam_num, att_texture, att_occ, random_glo, random_bkgd)
    else:
        pool = multiprocessing.Pool(processes=num_proc)
        for i in range(1,num_proc+1):
            pool.apply_async(samples_and_save, (i, texture_id, num_gpu, sam_num, att_texture, att_occ, random_glo, random_bkgd))
        pool.close()
        pool.join()

def collect_annotation(num_proc=40,sam_num=1,training_split=1,f_keep_training=False,root_path='exp',loop_num=-1):
    testing_split = sam_num-training_split
    
    len_label, len_label_test = 0,0
    path, path_test = [], []
    bbox, bbox_test = [], []
    pose, pose_test = [], []
    global_orient, global_orient_test = [], []
    beta, beta_test = [], []
    gender, gender_test = [], []

    for i in range(num_proc):
        npz_name = 'exps/Proc_'+str(i+1)+'/data_generated/generated_mmpose.npz'
        if os.path.exists(npz_name):
            annotation = np.load(npz_name,allow_pickle=True)
            
            path.append(annotation['image_path'][:training_split]); path_test.append(annotation['image_path'][training_split:])
            bbox.append(annotation['bbox_xywh'][:training_split]); bbox_test.append(annotation['bbox_xywh'][training_split:])
            pose.append(annotation['smpl'].item()['body_pose'][:training_split]); pose_test.append(annotation['smpl'].item()['body_pose'][training_split:])
            global_orient.append(annotation['smpl'].item()['global_orient'][:training_split]); global_orient_test.append(annotation['smpl'].item()['global_orient'][training_split:])
            beta.append(annotation['smpl'].item()['betas'][:training_split]); beta_test.append(annotation['smpl'].item()['betas'][training_split:])
            gender.append(annotation['meta'].item()['gender'][:training_split]); gender_test.append(annotation['meta'].item()['gender'][training_split:])

            len_label += training_split; len_label_test += testing_split

    path=np.concatenate(path); path_test=np.concatenate(path_test)
    bbox=np.concatenate(bbox); bbox_test=np.concatenate(bbox_test)
    pose=np.concatenate(pose); pose_test=np.concatenate(pose_test)
    global_orient=np.concatenate(global_orient); global_orient_test=np.concatenate(global_orient_test)
    beta=np.concatenate(beta); beta_test=np.concatenate(beta_test)
    gender=np.concatenate(gender); gender_test=np.concatenate(gender_test)

    __key_strict__ = annotation['__key_strict__']
    __temporal_len__ = np.array(len_label); __temporal_len_test__ = np.array(len_label_test)
    __keypoints_compressed__ = annotation['__keypoints_compressed__']
    config = annotation['config']
    smpl = annotation['smpl']; smpl_test = annotation['smpl']
    meta = annotation['meta']; meta_test = annotation['meta']

    if f_keep_training:
        if loop_num == 1:
            pass
        else:
            # load previous training data
            prefix = root_path+'/exps_loop_'+str(loop_num-1)+'/'
            data_before = np.load(prefix+'spin_adv_train.npz',allow_pickle=True)
            
            bef__temporal_len__ = data_before['__temporal_len__']
            
            bef_path_temp = data_before['image_path']
            bef_path = []
            for i in range(len(bef_path_temp)):
                if bef_path_temp[i][:4] == 'Proc':
                    bef_path.append('../' + prefix + bef_path_temp[i])
                else:
                    bef_path.append(bef_path_temp[i])
            bef_path=np.array(bef_path)
            
            bef_bbox = data_before['bbox_xywh']
            bef_pose = data_before['smpl'].item()['body_pose']
            bef_global_orient = data_before['smpl'].item()['global_orient']
            bef_beta = data_before['smpl'].item()['betas']
            bef_gender = data_before['meta'].item()['gender']

            # combine it with previous datas
            __temporal_len__ = np.array(__temporal_len__ + bef__temporal_len__)
            path = np.concatenate((bef_path,path),axis=0)
            bbox = np.concatenate((bef_bbox,bbox),axis=0)
            pose = np.concatenate((bef_pose,pose),axis=0)
            global_orient = np.concatenate((bef_global_orient,global_orient),axis=0)
            beta = np.concatenate((bef_beta,beta),axis=0)
            gender = np.concatenate((bef_gender,gender),axis=0)

    smpl.item()['body_pose']=pose; smpl_test.item()['body_pose']=pose_test
    smpl.item()['global_orient']=global_orient; smpl_test.item()['global_orient']=global_orient_test
    smpl.item()['betas']=beta; smpl_test.item()['betas']=beta_test
    meta.item()['gender']=gender; meta_test.item()['gender']=gender_test
    
    out_name = 'exps/spin_adv_train.npz'
    np.savez(out_name, __key_strict__=__key_strict__, __temporal_len__=__temporal_len__, __keypoints_compressed__=__keypoints_compressed__,
            image_path=path, bbox_xywh=bbox, config=config, smpl=smpl, meta=meta)
        
    out_name = 'exps/spin_adv_test.npz'
    np.savez(out_name, __key_strict__=__key_strict__, __temporal_len__=__temporal_len_test__, __keypoints_compressed__=__keypoints_compressed__,
            image_path=path_test, bbox_xywh=bbox_test, config=config, smpl=smpl_test, meta=meta_test)




if __name__ == '__main__':

    args = parse_args()
    
    num_proc = args.num_proc
    loop_num = args.loop_num
    root_path = args.root_name
    texture_id = args.texture_id
    num_gpu = args.num_gpu

    f_att_texture = False
    f_att_occ = False
    f_random_glo = True
    f_random_bkgd = True

    f_keep_training = False 

    assert f_att_texture == False 
    # Current released code does not support different textures. If you want to use different textures, you need to provide your own UV map. 
    # You can use the UV map from the Surreal project (https://github.com/gulvarol/surreal). Then you can set this True.
    assert f_att_occ == False
    # Current released code does not support occlusion. If you want to study occlusion, you need to provide your own patch occluder 
    # (see utils_PoseExaminer/render_pose.py, Line204, for deatils), then you can set this True.
    
    ### Step 2: sample 40*500 failure cases for training
    sample_poses(num_proc=num_proc,texture_id=texture_id,num_gpu=num_gpu,sam_num=1000,att_texture=f_att_texture, att_occ=f_att_occ, random_glo=f_random_glo, random_bkgd=f_random_bkgd)
    collect_annotation(num_proc=num_proc,sam_num=1000,training_split=500,f_keep_training=f_keep_training,root_path=root_path,loop_num=loop_num)


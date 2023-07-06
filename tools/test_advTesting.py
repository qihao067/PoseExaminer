import argparse
import os
import sys
sys.path.append('./utils_PoseExaminer/')

import torch
#### for adv testing
import multiprocessing
import time
import numpy as np

import utils_PoseExaminer.models.policy_2step as Policy
from utils_PoseExaminer.render_pose import PoseRender



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


def experiments(proc_id,texture_id=-1):

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
    std_limit = 2 #1.5
    texture_id = texture_id  # -1: mine texture
    with_bkground = False

    phase = 3 # evaluation
    iter = 9999
    
    ##########
    ########## initial smpl pose model
    num_poses = 200 # sample 200 poses
    uv_pose_render = PoseRender(texture_id=texture_id,num_poses=num_poses,proc_id=proc_id)
    
    from tools.run_mmhuman import PoseEva_mmhuman
    pose_eva = PoseEva_mmhuman(args=args, proc_id=proc_id)

    ##########
    ########## initial rl model
    def collate_fn(x):
        return x[0]

    policy = Policy.GaussianPolicy(1, num_dim=35, num_imgs=num_poses, std_limit=std_limit).to(device)

    ##### Run Learning Algorithm #####
    if True:
        start_time = time.time()
        print("--- [TIME] run session: %s seconds ---" % (time.time() - start_time))

        current_time = time.time()

        with torch.no_grad():
            sample_dim = None

            p_action, high, low = load_results(proc_id,device)
            
            print("--- [TIME] from policy to action : %s seconds ---" % (time.time() - current_time))
            current_time = time.time()
            
            # generate new test samples with current action
            multi_sampled_pose_body_mean, root_orient, border_log_probs, top, bottom, _useless = uv_pose_render.render(p_action, low=low, high=high, with_bkground=with_bkground, phase=phase, sample_dim=sample_dim)
            
            print("--- [TIME] Rendering : %s seconds ---" % (time.time() - current_time))
            current_time = time.time()
                
            # running pare
            mpjpe = 0.0
            rl_output_mpjpe_mean, rl_output_mpjpe_min, rl_output_mpjpe_max, rl_output_mpjpe_median, per_c = pose_eva.eva(proc_id,iter,output_ver='all')
                
            print("--- [TIME] Testing PARE : %s seconds ---" % (time.time() - current_time))
            current_time = time.time()

            # save all stuff
            print(' ### iter [{}] / 10*Gap [{}] / PARE_MPJPE(RL target) [{}] / OUTPUT_MPJPE(real error mean) [{}] / OUTPUT_MPJPE(real error min) [{}] / OUTPUT_MPJPE(real error max) [{}] / OUTPUT_MPJPE(real error median) [{}] / Per_C(percent of samples < 90) [{}]'
                    .format(iter, (high+low).mean().cpu().item()*10, mpjpe, rl_output_mpjpe_mean, rl_output_mpjpe_min, rl_output_mpjpe_max, rl_output_mpjpe_median, per_c))
            if proc_id==-1:
                result_save_file = "evaluation_error.txt"
            else:
                result_save_file = "exps/Proc_"+str(proc_id)+"/evaluation_error.txt"
            with open(result_save_file,"a") as f:
                f.write(' ### iter [{}] / NormGap [{}] / PARE_MPJPE(RL target) [{}] / OUTPUT_MPJPE(real error mean) [{}] / OUTPUT_MPJPE(real error min) [{}] / OUTPUT_MPJPE(real error max) [{}] / OUTPUT_MPJPE(real error median) [{}] / Per_C(percent of samples < 90) [{}] \n'
                    .format(iter, torch.norm(high+low).cpu().item(), mpjpe, rl_output_mpjpe_mean, rl_output_mpjpe_min, rl_output_mpjpe_max, rl_output_mpjpe_median, per_c))
            np.savez(save_path+'/policy_%.5d.npz'%iter,
                    noise_rand_top = high.detach().cpu().numpy(), noise_rand_bottom = low.detach().cpu().numpy(),
                    mean_pose = multi_sampled_pose_body_mean.cpu().numpy(), Zgen = p_action.cpu().numpy())


def parse_args():
    parser = argparse.ArgumentParser(description='mmhuman3d test model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('--work-dir', help='the dir to save evaluation results')
    parser.add_argument('--checkpoint', help='checkpoint file')
    parser.add_argument('--out', help='output result file')
    parser.add_argument('--metrics', type=str, nargs='+', default='pa-mpjpe', help='evaluation metric, which depends on the dataset')
    parser.add_argument('--gpu_collect', action='store_true', help='whether to use gpu to collect results')
    parser.add_argument('--tmpdir', help='tmp dir for writing some results')
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm', 'mpi'], default='none', help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--num_proc', type=int, default=40)
    parser.add_argument('--texture_id', type=int, default=-1)
    parser.add_argument('--num_gpu', type=int, default=8)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda', help='device used for testing')
    args = parser.parse_args()
    # if 'LOCAL_RANK' not in os.environ:
    #     os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def throw_error(e):
    raise e

if __name__ == '__main__':
    args=parse_args()

    num_proc = args.num_proc
    texture_id = args.texture_id

    multiprocessing.set_start_method("forkserver")

    INIT_TIME = time.time()

    for i in range(1,num_proc+1):
        os.rename('./exps/Proc_'+str(i), './exps/Proc_'+str(i)+'_training')
        os.makedirs('./exps/Proc_'+str(i)+'')
        os.makedirs('./exps/Proc_'+str(i)+'/checkpoints')
        os.makedirs('./exps/Proc_'+str(i)+'/data_generated')
        os.makedirs('./exps/Proc_'+str(i)+'/visulization')
    
    if num_proc == 1:
        experiments(1,texture_id)
    else:
        processes=[]
        for i in range(1,num_proc+1):
            p = multiprocessing.Process(target = experiments, args = (i,texture_id))
            processes.append(p)
        for t in processes:
            t.start()
        for t in processes:
            t.join()


        print ('Exiting Main Thread : '+str(time.time()-INIT_TIME))
    
    for i in range(1,num_proc+1):
        os.rename('./exps/Proc_'+str(i), './exps/Proc_'+str(i)+'_evaluation')

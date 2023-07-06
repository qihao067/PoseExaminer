import argparse
import os
import sys
sys.path.append('./utils_PoseExaminer/')

import torch
import multiprocessing
import time
import numpy as np

import utils_PoseExaminer.models.policy_2step as Policy
from utils_PoseExaminer.render_pose import PoseRender
from utils_PoseExaminer.contraints import JOINT_RANGE


def experiments(proc_id,texture_id=-1):

    args=parse_args()

    time.sleep((proc_id-1)*2)

    num_gpu = args.num_gpu
    gpu_id = proc_id % num_gpu
    # gpu_list=['0','1','2','3','4','5','6','7']
    gpu_list = list(range(0, num_gpu))
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_list[gpu_id])
    ##############

    args = parse_args()
    loop_num = args.loop_num

    save_path = './exps/Proc_'+str(proc_id)+'/checkpoints'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Params
    max_iters = 600 # 300(shape) # 500(is good enough)
    policy_lr = 2e-1 # 5e-2(skinColor) # 1e-1(shape) # 2e-1(Ori)
    boundary_lr = 1e-4 # 1e-4 (PARE/SPIN)  # 3e-4 (HMR)
    baseline_alpha = 0.5
    std_limit = 2 # 1.5
    texture_id = texture_id  # -1
    with_bkground = False
    phase_iter = 300 # 250 #200
    mpjpe_threshold = 90
    
    # step size
    max_gap = 0.05
    min_gap = 0.005
    max_mpj = 120
    min_mpj = 95
    ini_gap = 0.03
    current_step = 0.00 # it is actually useless, it is just used for initization

    sample_num = 5
    sample_iter = 3

    consider_shape = False
    consider_skin_color = False

    ##########
    ########## initial smpl pose model
    num_poses = 50
    uv_pose_render = PoseRender(texture_id=texture_id,num_poses=num_poses,proc_id=proc_id)
    
    from tools.run_mmhuman import PoseEva_mmhuman 
    pose_eva = PoseEva_mmhuman(args=args, proc_id=proc_id)

    ##########
    ########## initial rl model
    def collate_fn(x):
        return x[0]
    
    if consider_shape:
        policy = Policy.GaussianPolicy(1, num_dim=10, num_imgs=num_poses, std_limit=std_limit, ini_gap=ini_gap).to(device)
        sample_pose = torch.from_numpy(np.random.normal(0., 1., size=(1, 32)).astype(np.float32)).to(device)
    elif consider_skin_color:
        policy = Policy.GaussianPolicy(1, num_dim=3, num_imgs=num_poses, std_limit=std_limit, ini_gap=ini_gap).to(device)
        sample_pose = torch.from_numpy(np.random.normal(0., 1., size=(1, 32)).astype(np.float32)).to(device)
    else:
        policy = Policy.GaussianPolicy(1, num_dim=32, num_imgs=num_poses, std_limit=std_limit, ini_gap=ini_gap).to(device)
    

    policy_optimizer = torch.optim.Adam(policy.parameters(), lr = policy_lr)
    border_optimizer = torch.optim.SGD([policy.noise_rand_top,policy.noise_rand_bottom], lr = boundary_lr, momentum=0.0)

    baseline = torch.FloatTensor(1).to(device)
    baseline[0] = 0.4

    sample_dim_list = np.array([x for x in range(126)])
    sampe_p_list = np.ones(126)/126
    ##### Run Learning Algorithm #####
    if True:
        mpjpe_record = []

        start_time = time.time()
        print("--- [TIME] run session: %s seconds ---" % (time.time() - start_time))

        joint_range = torch.from_numpy(np.array(JOINT_RANGE).T * np.pi).to(device)
        for iter in range(max_iters):
            current_time = time.time()
            # Generate new parameters with policy
            fixed_input = torch.ones(1).view(1,-1).to(device)

            if iter<phase_iter:
                phase = 1
                sample_dim = None
            else:
                phase = 2
                for param in policy.fc_action.parameters():
                    param.requires_grad = False 
            
                if (iter-phase_iter)%sample_iter == 0:
                    sample_id = np.random.choice(a=sample_dim_list, size=sample_num, replace=False, p=sampe_p_list)
                sample_dim = np.zeros(126)
                sample_dim[sample_id] = 1.0
                sample_dim=sample_dim.reshape(2,-1)

            p_action, action_log_probs, gap, _p_mean, _p_std, low, high = policy.act(fixed_input, phase=phase)
            if consider_shape or consider_skin_color:
                p_action = torch.cat([sample_pose,p_action],1)

            print("--- [TIME] from policy to action : %s seconds ---" % (time.time() - current_time))
            current_time = time.time()
            
            # generate new test samples with current action
            multi_sampled_pose_body_mean, root_orient, border_log_probs, top, bottom, constraint_zgen = uv_pose_render.render(p_action, low=low, high=high, with_bkground=with_bkground, phase=phase, sample_dim=sample_dim, step_gap=current_step)
            
            print("--- [TIME] Rendering : %s seconds ---" % (time.time() - current_time))
            current_time = time.time()
                
            with torch.no_grad():
                # running pare
                mpjpe, rl_output_mpjpe, rl_output_permpjpe_smpl = pose_eva.eva(proc_id,iter,output_ver='min')
                
                if phase == 1:
                    dist = 50 - mpjpe * 0.1
                elif phase == 2:
                    dist = max(0.0,(rl_output_mpjpe-mpjpe_threshold)*gap.mean()*20)
                else:
                    dist = 0
                    import pdb; pdb.set_trace()

                print("--- [TIME] Testing PARE : %s seconds ---" % (time.time() - current_time))
                current_time = time.time()

            ##### Train RL policy #####
            if phase==2:
                adv = torch.tensor(dist).to(device)
            else:
                adv = (dist - baseline).detach().to(device)

            if phase == 1:
                policy_loss = (action_log_probs * adv).mean()
                policy_optimizer.zero_grad() 

                policy_loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), 5.0) 
                policy_optimizer.step() 

                # update step gap: # although it is computed in very step, but is only used in the final step
                if rl_output_mpjpe>max_mpj:
                    current_step = max_gap
                elif rl_output_mpjpe<min_mpj:
                    current_step = min_gap
                else:
                    current_step = (max_gap-min_gap)/(max_mpj-min_mpj)*(rl_output_mpjpe-min_mpj) + min_gap
            
            elif phase == 2:
                weights = sample_dim[0]+sample_dim[1]
                weights = torch.from_numpy(weights).to(device)
                # 
                weights_joint_range = (1.0*(top<joint_range[0]))*(1.0*(bottom>joint_range[1]))
                weights_joint_range = weights_joint_range.squeeze()
                # 
                weight_zgen = constraint_zgen.clone()

                # 
                sample_p_joint_range = torch.cat(((1.0*(top<joint_range[0])),(1.0*(bottom>joint_range[1]))),1).cpu().numpy()[0]
                sample_p_joint_range[sample_p_joint_range==0]=0.5
                sampe_p_list = sample_p_joint_range * sampe_p_list
                sampe_p_list = sampe_p_list/sampe_p_list.sum()

                border_log_probs_w = torch.sum((border_log_probs*weights*weights_joint_range*weight_zgen), dim=1 , keepdim=True)
                policy_loss = (border_log_probs_w * adv).mean()
                    
                if policy_loss.isnan().sum() != 0: 
                    raise NotImplementedError("NAN exists : train_RLpose - 1")

                # update step gap:
                if rl_output_mpjpe>max_mpj:
                    current_step = max_gap
                elif rl_output_mpjpe<min_mpj:
                    current_step = min_gap
                else:
                    current_step = (max_gap-min_gap)/(max_mpj-min_mpj)*(rl_output_mpjpe-min_mpj) + min_gap
                
                border_optimizer.zero_grad() 
                policy_loss.backward(retain_graph=True)
                # modify the grad based on MPJPE
                grad_step = -policy.noise_rand_top.grad.min().item()*boundary_lr
                if 0.0-policy.noise_rand_top.grad.min().item() < 1.3*current_step/boundary_lr and 0-policy.noise_rand_top.grad.min().item() > 0.7*current_step/boundary_lr:
                    pass
                else:
                    temp_grad = -max(min(0-policy.noise_rand_top.grad.min().item(),1.3*current_step/boundary_lr),0.7*current_step/boundary_lr)
                    policy.noise_rand_top.grad[policy.noise_rand_top.grad!=0] = temp_grad
                    policy.noise_rand_bottom.grad[policy.noise_rand_bottom.grad!=0] = temp_grad
                
                border_optimizer.step()
                    

            if phase==2:
                baseline = torch.tensor(0.0).to(device)
            else:
                baseline = torch.tensor([baseline.detach() * (1 - baseline_alpha) + dist * baseline_alpha]).to(device)

            print("--- [TIME] RL - training : %s seconds ---" % (time.time() - current_time))
            current_time = time.time()

            # save all stuff
            if phase==2:
                print(' ### iter [{}] / Baseline [{:.4f}] / Embedding distance  [{}] / Policy_loss [{}] / 10*Gap [{}] / PARE_MPJPE(RL target) [{}] / OUTPUT_MPJPE(real error) [{}] / InvVposer(weight_zgen) [{}] / timeNeeds [{:.4f}]'
                        .format(iter, baseline.cpu().item() , dist, policy_loss.cpu().item(), gap.mean().cpu().item()*10, mpjpe, rl_output_mpjpe, weight_zgen.sum().cpu().item(), (time.time() - start_time)/(iter+1)*(max_iters-iter)/60/60))
                if proc_id==-1:
                    result_save_file = "training_loss.txt"
                else:
                    result_save_file = "exps/Proc_"+str(proc_id)+"/training_loss.txt"
                with open(result_save_file,"a") as f:
                    f.write(' ### iter [{}] / Baseline [{:.4f}] / Embedding distance  [{:.4f}] / Policy_loss [{:.4f}] / 10*Gap [{:.4f}] / PARE_MPJPE(RL target) [{:.4f}] / OUTPUT_MPJPE(real error) [{:.4f}] / InvVposer(weight_zgen) [{:.4f}]  / InvVposer+JointRange(weight) [{:.4f}] / grad[{:.4}] / grad_step[{:.4}] / mpjpe_step[{:.4}] / real_step[{:.8}] / timeNeeds [{:.4f}] \n'
                        .format(iter, baseline.cpu().item() , dist, policy_loss.cpu().item(), gap.mean().cpu().item()*10, mpjpe, rl_output_mpjpe, weight_zgen.sum().cpu().item(), (weights_joint_range * weight_zgen).sum().cpu().item(), policy.noise_rand_top.grad.min().item(), grad_step, current_step, policy.noise_rand_top.grad.min().item()*boundary_lr, (time.time() - start_time)/(iter+1)*(max_iters-iter)/60/60 ))
            else:
                print(' ### iter [{}] / Baseline [{:.4f}] / Embedding distance  [{}] / Policy_loss [{}] / 10*Gap [{}] / PARE_MPJPE(RL target) [{}] / OUTPUT_MPJPE(real error) [{}] / timeNeeds [{:.4f}]'
                        .format(iter, baseline.cpu().item() , dist, policy_loss.cpu().item(), gap.mean().cpu().item()*10, mpjpe, rl_output_mpjpe, (time.time() - start_time)/(iter+1)*(max_iters-iter)/60/60))
                if proc_id==-1:
                    result_save_file = "training_loss.txt"
                else:
                    result_save_file = "exps/Proc_"+str(proc_id)+"/training_loss.txt"
                with open(result_save_file,"a") as f:
                    f.write(' ### iter [{}] / Baseline [{:.4f}] / Embedding distance  [{:.4f}] / Policy_loss [{:.4f}] / 10*Gap [{:.4f}] / PARE_MPJPE(RL target) [{:.4f}] / OUTPUT_MPJPE(real error) [{:.4f}] / timeNeeds [{:.4f}] \n'
                        .format(iter, baseline.cpu().item() , dist, policy_loss.cpu().item(), gap.mean().cpu().item()*10, mpjpe, rl_output_mpjpe, (time.time() - start_time)/(iter+1)*(max_iters-iter)/60/60))
            #
            np.savez(save_path+'/policy_%.5d.npz'%iter, mean = _p_mean.detach().cpu().numpy(), std = _p_std.detach().cpu().numpy(),
                    noise_rand_top = policy.noise_rand_top.detach().cpu().numpy(), noise_rand_bottom = policy.noise_rand_bottom.detach().cpu().numpy(),
                    mean_pose = multi_sampled_pose_body_mean.cpu().numpy(), root_orient=root_orient.cpu().numpy() ,Zgen = p_action.cpu().numpy())
            
            # 
            if consider_shape or consider_skin_color:
                pass
            else:
                mpjpe_record.append(rl_output_mpjpe)
                if len(mpjpe_record) > 20:
                    if np.array(mpjpe_record[-10:]).min()>mpjpe_threshold+10:
                        phase_iter = iter + 1
            
            # 
            if iter==phase_iter-1 and rl_output_mpjpe<mpjpe_threshold and phase_iter>100: 
                break
            
        torch.save(policy.state_dict(),'exps/Proc_'+str(proc_id)+'/Policy_checkpoints.pt')



def parse_args():
    parser = argparse.ArgumentParser(description='mmhuman3d test model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('--work-dir', help='the dir to save evaluation results')
    parser.add_argument('--checkpoint', help='checkpoint file')
    parser.add_argument('--loop_num', type=int, default=-1)
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
        os.makedirs('./exps/Proc_'+str(i))
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

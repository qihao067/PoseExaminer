import numpy as np
import sys

def load_and_conclude(num_proc,ckp_name):
    GAP = []
    MPJPE_STEP1 = []
    MPJPE_MEAN = []
    MPJPE_MIN = []
    MPJPE_MAX = []
    MPJPE_MEDIAN = []
    PER_C = []
    cont = 0.0
    for proc_id in range(num_proc):
        ### read step 1 results
        f = open('./exps/Proc_'+str(proc_id+1)+'_training/training_loss.txt','r')
        lines = f.readlines()
        # 
        phase_1 = 0
        while phase_1<len(lines) and len(lines[phase_1])<300:
            phase_1+=1
        phase_1-=1

        try:
            line_sel = lines[phase_1]
        except:
            line_sel = lines[-1]
        line = line_sel.split('[')
        MPJPE_STEP1.append(float(line[7].split(']')[0]))
        with open('./exps/final_results.txt',"a") as f:
            f.write('### Proc '+ str(proc_id+1) +'_STEP1_'+line_sel)

        ### read step 2 results
        f = open('./exps/Proc_'+str(proc_id+1)+'_evaluation/evaluation_error.txt','r')
        lines = f.readlines()
        line = lines[0]
        line = line.split('[')
        if float(line[2].split(']')[0])>0.6:
            GAP.append(float(line[2].split(']')[0]))
            MPJPE_MEAN.append(float(line[4].split(']')[0]))
            MPJPE_MIN.append(float(line[5].split(']')[0]))
            MPJPE_MAX.append(float(line[6].split(']')[0]))
            MPJPE_MEDIAN.append(float(line[7].split(']')[0]))
            PER_C.append(float(line[8].split(']')[0]))

        with open('./exps/final_results.txt',"a") as f:
            f.write('### Proc '+ str(proc_id+1) +lines[0]+ '\n')
        
        if float(line[2].split(']')[0])>0.6:
            cont+=1
                
    with open('./exps/final_results.txt',"a") as f:
        f.write('\n \n ')
        f.write('###'*40)
        f.write('\n \n ')
        f.write(' ### Final / MPJPE_STEP1 [{}] / NormGap-MEAN [{}] / mean_mpjpe-MEAN [{}] / min_mpjpe-MEAN [{}] / max_mpjpe-MEAN [{}] / median_mpjpe-MEAN [{}] / Pre_c-MEAN [{}] / SuccRate [{}]\n'
            .format(np.array(MPJPE_STEP1).mean(),np.array(GAP).mean(),np.array(MPJPE_MEAN).mean(),np.array(MPJPE_MIN).mean(),np.array(MPJPE_MAX).mean(),np.array(MPJPE_MEDIAN).mean(),np.array(PER_C).mean(),cont/num_proc))

    with open('evaluation_results.txt',"a") as f:
        f.write('###'+'\n')
        f.write('### Robustness testing on checkpoints : [ {} ]\n'.format(ckp_name) )
        f.write('### Final / MPJPE_STEP1 [{}] / NormGap-MEAN [{}] / mean_mpjpe-MEAN [{}] / min_mpjpe-MEAN [{}] / max_mpjpe-MEAN [{}] / median_mpjpe-MEAN [{}] / Pre_c-MEAN [{}] / SuccRate [{}]\n'
            .format(np.array(MPJPE_STEP1).mean(),np.array(GAP).mean(),np.array(MPJPE_MEAN).mean(),np.array(MPJPE_MIN).mean(),np.array(MPJPE_MAX).mean(),np.array(MPJPE_MEDIAN).mean(),np.array(PER_C).mean(),cont/num_proc))
        f.write('###'*40+'\n')
        f.write('\n \n \n')

if __name__ == '__main__':

    num_proc = int(sys.argv[1])
    ckp_name = sys.argv[2]
    
    load_and_conclude(num_proc,ckp_name)

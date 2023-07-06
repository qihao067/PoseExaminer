import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn.init as init

def ppo_loss(adv, action_log_probs, old_action_log_probs, clip_param):
    """
    This is the Proximal Policy Optimization loss.
    :FloatTensor adv: advantage estimation
    :Variable action_log_probs: pi_{theta}(rho, phi)
    :FloatTensor old_action_log_probs: pi_{theta_old}(rho, phi)
    :float clip_param: clip parameter
    :return: ppo_loss
    """
    ratio = torch.exp(action_log_probs - old_action_log_probs)
    surr1 = ratio * adv
    surr2 = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * adv

    # PPO's pessimistic surrogate (L^CLIP)
    action_loss = -torch.min(surr1, surr2).mean() 

    return action_loss

def load_filtered_state_dict(model, snapshot):
    model_dict = model.state_dict()
    snapshot = {k: v for k, v in snapshot.items() if k in model_dict}
    model_dict.update(snapshot)
    model.load_state_dict(model_dict)
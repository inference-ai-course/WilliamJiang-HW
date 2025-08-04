#python code

import torch
import torch.nn.functional as F

# Simulated log-probs of chosen vs rejected completions
chosen_logp = torch.tensor([[-1.0]])
rejected_logp = torch.tensor([[-2.0]])

def dpo_loss(chosen_logp, rejected_logp, beta=0.1):
    return -F.logsigmoid((chosen_logp - rejected_logp) / beta).mean()

print("DPO Loss:", dpo_loss(chosen_logp, rejected_logp).item())
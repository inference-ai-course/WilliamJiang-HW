import torch
import torch.nn.functional as F

# Simulated old and new policy log-probs (log π_θ(a|s) and log π_θ_old(a|s))
old_log_prob = torch.tensor([[-1.0]])  # from reference policy (e.g. GPT-4 before PPO step)
new_log_prob = torch.tensor([[-0.8]])  # from updated policy
reward = torch.tensor([[1.0]])         # reward from human or reward model
epsilon = 0.2                          # PPO clipping parameter

# Compute ratio of new to old policy
log_ratio = new_log_prob - old_log_prob
ratio = torch.exp(log_ratio)

# Unclipped and clipped advantages
advantage = reward  # assume reward ~ advantage for simplicity
clipped_ratio = torch.clamp(ratio, 1 - epsilon, 1 + epsilon)

# PPO loss (negative of the clipped surrogate objective)
ppo_loss = -torch.min(ratio * advantage, clipped_ratio * advantage).mean()
print("PPO Loss:", ppo_loss.item())
import numpy as np
from sgan.data.loader import data_loader
import os
import torch
from attrdict import AttrDict
from sgan.models import TrajectoryGenerator

num = 4
epoch = 200000
model_path = "checkpoints/checkpoint%d_fixed2_with_model_%d.pt" % (num, epoch)
plot_dict = {
    "x": [],
    "y1": [],
    "y2": []
}
plot_path = "plots/checkpoint%d_fixed2_%d_round_2.pt" % (num, epoch)
checkpoint = torch.load(model_path)
print("Loading succeed")

args = AttrDict(checkpoint['args'])
generator = TrajectoryGenerator(
        obs_len=args.obs_len,
        pred_len=args.pred_len,
        embedding_dim=args.embedding_dim,
        encoder_h_dim=args.encoder_h_dim_g,
        decoder_h_dim=args.decoder_h_dim_g,
        mlp_dim=args.mlp_dim,
        num_layers=args.num_layers,
        noise_dim=args.noise_dim,
        noise_type=args.noise_type,
        noise_mix_type=args.noise_mix_type,
        pooling_type=args.pooling_type,
        pool_every_timestep=args.pool_every_timestep,
        dropout=args.dropout,
        bottleneck_dim=args.bottleneck_dim,
        neighborhood_size=args.neighborhood_size,
        grid_size=args.grid_size,
        batch_norm=args.batch_norm)
generator.load_state_dict(checkpoint['g_state'])
generator.to(torch.device('cuda'))
generator.eval()
args.batch_size = 1
sets = [num]
_, loader = data_loader(args, sets)
count = 0
with torch.no_grad():
    for batch in loader:
        obs_traj, obs_traj_rel, ground_truth_list, mask_list, render_list, seq_start_end = batch
        obs_traj = obs_traj.cuda()
        obs_traj_rel = obs_traj_rel.cuda()
        outputs, times = generator(obs_traj, obs_traj_rel, seq_start_end)
        for i in range(len(outputs)):
            output = outputs[i]
            ground_truth, mask = ground_truth_list[i], mask_list[i]
            ground_truth = torch.tensor(ground_truth).cuda()
            mask = torch.tensor(mask).cuda()
            future_mask = mask > 0
            if (torch.sum(future_mask) == 0):
                continue
            count += 1
            norm_ground_truth = torch.sum(torch.linalg.norm(ground_truth, dim = 2)*future_mask) / torch.sum(future_mask) 
            norm_diff = torch.sum(torch.linalg.norm(output[:, :, 0:2] - ground_truth, dim = 2)*future_mask) / torch.sum(future_mask)
            norm_uncertainty = torch.sum(torch.linalg.norm(output[:, :, 2:4], dim = 2)*future_mask) / torch.sum(future_mask)
            plot_dict["x"].append(norm_ground_truth.cpu())
            plot_dict["y1"].append(norm_diff.cpu())
            plot_dict["y2"].append(norm_uncertainty.cpu())
        if (count % 1000 == 0):
            print(norm_ground_truth, norm_diff, norm_uncertainty)
torch.save(plot_dict, plot_path)

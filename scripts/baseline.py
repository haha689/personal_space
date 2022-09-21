from inference import SGANInference
from inference import computeDistances
from inference import get_model_matrix
import numpy as np
from sgan.data.loader import data_loader
import os
import torch

model_name = "zara2_8_model"
model_path = "models/sgan-models/%s.pt" % model_name
trajPlan = SGANInference(model_path)
generator = trajPlan.generator
_args = trajPlan.args
_args.batch_size = 1
sets = [0]
_, loader = data_loader(_args, sets)
num_of_predictions = 5 #obs_len = 8
checkpoint = {
    'ground_truth': [],
    'mask':[],
    'future_mask': [],
    'model_matrix': [],
    'mse_metric': [],
    'interaction_time':[],
    'delta':[],
    'predictions':[]
}
checkpoint_path = os.path.join(
    os.getcwd(), 'checkpoints', '%s_baseline_new.pt' % model_name
                )
print_every = 20
iter = 0

for batch in loader:
    obs_traj, obs_traj_rel, ground_truth_list, mask_list, render_list, seq_start_end = batch
    ground_truth, mask, render, seq_start_end = ground_truth_list[0], mask_list[0], render_list[0], seq_start_end[0] #batch_size = 1
    #print(ground_truth)
    future_mask = mask > 0
    if np.sum(future_mask) == 0:
        #print('not good batch')
        continue
    obs_traj = obs_traj.numpy()
    traj_length = len(obs_traj[1])
    matrices = []
    predicted = obs_traj.transpose((1,0,2))
    predictions = []
    for j in range(num_of_predictions):
        sub_matrices = np.zeros((len(predicted),len(predicted),2))
        history = predicted
        predicted = trajPlan.evaluate(history)
        predictions.append(predicted.transpose((1,0,2)))
        for x in range(len(predicted)):
            cors = computeDistances(predicted,x) #adjancey matrix
            cors = np.array(cors)
            sub_matrices[x,:,:] = cors
        #print('sub matrices')
        #print(sub_matrices)
        matrices.append(sub_matrices)
    matrices = np.array(matrices)
    predictions = np.concatenate(predictions, 0)
    delta = np.zeros((len(mask), len(mask), 2))
    for i in range (len(mask)):
        for j in range (len(mask)):
            delta_t = int(mask[i, j])
            if (delta_t > 0 and delta_t <= 40):
                coord_i = predictions[delta_t - 1, i, :]
                coord_j = predictions[delta_t - 1, j, :]
                delta[i, j, :] = coord_i - coord_j
    
    model_matrix = get_model_matrix(matrices)
    #print(model_matrix)
    mse_metric = np.sum(np.linalg.norm(model_matrix - ground_truth, axis = 2)*future_mask) / np.sum(future_mask)
    diff_norm = np.sum(np.linalg.norm(delta - ground_truth, axis = 2)*future_mask) / np.sum(future_mask)
    #print(mse_metric)
    #IF MASK is
    checkpoint['ground_truth'].append(ground_truth)
    checkpoint['mask'].append(mask)
    checkpoint['future_mask'].append(future_mask)
    checkpoint['model_matrix'].append(model_matrix)
    checkpoint['mse_metric'].append(mse_metric)
    checkpoint['delta'].append(delta)
    checkpoint['predictions'].append(predictions)
    torch.save(checkpoint, checkpoint_path)
    if (iter % print_every == 0):
        print(mse_metric)
    iter += 1




    
#print(trajPlan)

#take 1 batch
#make x number of predictions, find adjancey matrix and make a list of them
#find the best matrix out of them all (minimum)
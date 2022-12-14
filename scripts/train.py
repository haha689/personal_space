import argparse
import gc
import logging
from operator import matmul
import os
import sys
import time
import numpy as np

from collections import defaultdict

import torch
import torch.nn as nn
import torch.optim as optim

from sgan.data.loader import data_loader
from sgan.losses import gan_g_loss, gan_d_loss, l2_loss
from sgan.losses import displacement_error, final_displacement_error

from sgan.models import TrajectoryGenerator
from sgan.utils import int_tuple, bool_flag, get_total_norm
from sgan.utils import relative_to_abs, get_dset_path

torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser()
FORMAT = '[%(levelname)s: %(filename)s: %(lineno)4d]: %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT, stream=sys.stdout)
logger = logging.getLogger(__name__)

# Dataset options
parser.add_argument('--dataset_name', default='zara1', type=str)
parser.add_argument('--delim', default=' ')
parser.add_argument('--loader_num_workers', default=4, type=int)
parser.add_argument('--obs_len', default=8, type=int)
parser.add_argument('--pred_len', default=8, type=int)
parser.add_argument('--skip', default=1, type=int)

# Optimization
parser.add_argument('--batch_size', default=8, type=int)
parser.add_argument('--num_iterations', default=10000, type=int)
parser.add_argument('--num_epochs', default=100, type=int)

# Model Options
parser.add_argument('--embedding_dim', default=64, type=int)
parser.add_argument('--num_layers', default=1, type=int)
parser.add_argument('--dropout', default=0, type=float)
parser.add_argument('--batch_norm', default=0, type=bool_flag)
parser.add_argument('--mlp_dim', default=1024, type=int)

# Generator Options
parser.add_argument('--encoder_h_dim_g', default=64, type=int)
parser.add_argument('--decoder_h_dim_g', default=128, type=int)
parser.add_argument('--noise_dim', default=(0, ), type=int_tuple)
parser.add_argument('--noise_type', default='gaussian')
parser.add_argument('--noise_mix_type', default='ped')
parser.add_argument('--clipping_threshold_g', default=0, type=float)
parser.add_argument('--g_learning_rate', default=1e-5, type=float)
parser.add_argument('--g_steps', default=1, type=int)

# Pooling Options
parser.add_argument('--pooling_type', default='pool_net')
parser.add_argument('--pool_every_timestep', default=1, type=bool_flag)

# Pool Net Option
parser.add_argument('--bottleneck_dim', default=1024, type=int)

# Social Pooling Options
parser.add_argument('--neighborhood_size', default=2.0, type=float)
parser.add_argument('--grid_size', default=8, type=int)

# Discriminator Options
parser.add_argument('--d_type', default='local', type=str)
parser.add_argument('--encoder_h_dim_d', default=64, type=int)
parser.add_argument('--d_learning_rate', default=1e-5, type=float)
parser.add_argument('--d_steps', default=2, type=int)
parser.add_argument('--clipping_threshold_d', default=0, type=float)

# Loss Options
parser.add_argument('--l2_loss_weight', default=0, type=float)
parser.add_argument('--best_k', default=1, type=int)

# Output
parser.add_argument('--output_dir', default=os.getcwd())
parser.add_argument('--print_every', default=5, type=int)
parser.add_argument('--checkpoint_every', default=10000, type=int)
parser.add_argument('--checkpoint_name', default='checkpoint')
parser.add_argument('--checkpoint_start_from', default=None)
parser.add_argument('--restore_from_checkpoint', default=1, type=int)
parser.add_argument('--num_samples_check', default=5000, type=int)

# Misc
parser.add_argument('--use_gpu', default=1, type=int)
parser.add_argument('--timing', default=0, type=int)
parser.add_argument('--gpu_num', default="0", type=str)


def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight)


def get_dtypes(args):
    long_dtype = torch.LongTensor
    float_dtype = torch.FloatTensor
    if args.use_gpu == 1:
        long_dtype = torch.cuda.LongTensor
        float_dtype = torch.cuda.FloatTensor
    return long_dtype, float_dtype


def main(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_num
    # train_path = get_dset_path(args.dataset_name, 'train')
    # val_path = get_dset_path(args.dataset_name, 'val')

    long_dtype, float_dtype = get_dtypes(args)

    logger.info("Initializing train dataset")
    train_path = [0, 1, 2, 3]
    val_path = [4]
    train_dset, train_loader = data_loader(args, train_path)
    logger.info("Initializing val dataset")
    _, val_loader = data_loader(args, val_path)

    iterations_per_epoch = len(train_dset) / args.batch_size / args.d_steps
    if args.num_epochs:
        args.num_iterations = int(iterations_per_epoch * args.num_epochs)

    logger.info(
        'There are {} iterations per epoch'.format(iterations_per_epoch)
    )

    print(args.pooling_type)
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


    generator.apply(init_weights)
    generator.type(float_dtype).train()
    logger.info('Here is the generator:')
    logger.info(generator)


    optimizer_g = optim.Adam(generator.parameters(), lr=args.g_learning_rate)

    # Maybe restore from checkpoint
    restore_path = None
    if args.checkpoint_start_from is not None:
        restore_path = args.checkpoint_start_from
    elif args.restore_from_checkpoint == 1:
        restore_path = os.path.join(args.output_dir, 'checkpoints',
                                    '%s_with_model.pt' % args.checkpoint_name)

    if restore_path is not None and os.path.isfile(restore_path):
        logger.info('Restoring from checkpoint {}'.format(restore_path))
        checkpoint = torch.load(restore_path)
        generator.load_state_dict(checkpoint['g_state'])
       
        optimizer_g.load_state_dict(checkpoint['g_optim_state'])
       
        t = checkpoint['counters']['t']
        epoch = checkpoint['counters']['epoch']
        checkpoint['restore_ts'].append(t)
    else:
        # Starting from scratch, so initialize checkpoint data structure
        t, epoch = 0, 0
        checkpoint = {
            'args': args.__dict__,
            'G_losses':[],
            
            'losses_ts': [],
            'metrics_val': [],
            'metrics_train': [],
            'sample_ts': [],
            'restore_ts': [],
            'norm_g': [],
            'mse_metric_val': [],
            'mse_metric_train': [],
           
            'counters': {
                't': None,
                'epoch': None,
            },
            'g_state': None,
            'g_optim_state': None
        }
    t0 = None

    while t < args.num_iterations:
        gc.collect()
        epoch += 1
        logger.info('Starting epoch {}'.format(epoch))
        for batch in train_loader:
            if args.timing == 1:
                torch.cuda.synchronize()
                t1 = time.time()

            # Decide whether to use the batch for stepping on discriminator or
            # generator; an iteration consists of args.d_steps steps on the
            # discriminator followed by args.g_steps steps on the generator.
            
        
            step_type = 'g'
            losses_g = generator_step(args, batch, generator,
                                        optimizer_g)
            checkpoint['norm_g'].append(
                get_total_norm(generator.parameters())
            )

             
            if args.timing == 1:
                torch.cuda.synchronize()
                t2 = time.time()
                logger.info('{} step took {}'.format(step_type, t2 - t1))
          
            if args.timing == 1:
                if t0 is not None:
                    logger.info('Interation {} took {}'.format(
                        t - 1, time.time() - t0
                    ))
                t0 = time.time()

            # Maybe save loss
            if t % args.print_every == 0:
                logger.info('t = {} / {}'.format(t + 1, args.num_iterations))
                
                logger.info('  [loss]: {:.3f}'.format(losses_g.item()))
                checkpoint['G_losses'].append(losses_g.item())
                checkpoint['losses_ts'].append(t)

            #''' Maybe save a checkpoint
            if t > 0 and t % args.checkpoint_every == 0:
                checkpoint['counters']['t'] = t
                checkpoint['counters']['epoch'] = epoch
                checkpoint['sample_ts'].append(t)

                # Check stats on the validation set
                logger.info('Checking stats on val ...')
                metrics_val, mse_metric_val = check_accuracy(
                    args, val_loader, generator)
                logger.info('Checking stats on train ...')
                metrics_train, mse_metric_train = check_accuracy(
                    args, train_loader, generator)

                
                logger.info('  [val] : {:.3f}'.format(metrics_val))
                checkpoint['metrics_val'].append(metrics_val)
                checkpoint['mse_metric_val'].append(mse_metric_val)
               
                logger.info('  [train] : {:.3f}'.format(metrics_train))
                checkpoint['metrics_train'].append(metrics_train)
                checkpoint['mse_metric_train'].append(mse_metric_train)

                # Save another checkpoint with model weights and
                # optimizer state
                checkpoint['g_state'] = generator.state_dict()
                checkpoint['g_optim_state'] = optimizer_g.state_dict()
                checkpoint_path = os.path.join(
                    args.output_dir, 'checkpoints', '%s_with_model_%d.pt' % (args.checkpoint_name, t)
                )
                logger.info('Saving checkpoint to {}'.format(checkpoint_path))
                torch.save(checkpoint, checkpoint_path)

                logger.info('Done.')

            t += 1
            if t >= args.num_iterations:
                break

        print(losses_g.item())

# MLE loss for 1d outputs
def loss_func_1d(output, ground_truth, mask):
    pred = output[:, :, 0]
    std = torch.exp(output[:, :, 1])
    diff = pred - ground_truth

    epsilon = 1e-10

    loss_1 = torch.sqrt(2 * np.pi * std ** 2)
    loss_2 = - diff ** 2 / (2 * std ** 2)

    loss = -(-torch.log(torch.clamp(loss_1, min=epsilon)) + loss_2)

    # make sure stds are positive
    #loss += (torch.abs(std) - std) ** 2

    return torch.sum(loss * mask) / torch.sum(mask)

# MLE loss for 2d outputs
def loss_func_2d(output, ground_truth, mask):
    pred_x = output[:, :, 0]
    pred_y = output[:, :, 1]
    std_x = torch.exp(output[:, :, 2])
    std_y = torch.exp(output[:, :, 3])
    cov = torch.tanh(output[:, :, 4])
    true_x = ground_truth[:, :, 0]
    true_y = ground_truth[:, :, 1]
    diff_x = pred_x - true_x 
    diff_y = pred_y - true_y

    epsilon = 1e-10

    loss_1 = 2 * np.pi * std_x * std_y * torch.sqrt(1 - cov ** 2)
    z = diff_x ** 2 / (std_x ** 2) + diff_y ** 2 / (std_y ** 2) \
        - (2 * cov * diff_x * diff_y) / (std_x * std_y)
    loss_2 = - torch.clamp(z, min=epsilon) / (2 * (1 - cov ** 2) + epsilon)
    
    loss = -(-torch.log(torch.clamp(loss_1, min=epsilon)) + 
             loss_2)

    mse_metric = torch.sum(torch.linalg.norm(output[:, :, 0:2] - ground_truth, dim = 2)*mask) / torch.sum(mask)

    # make sure stds are positive
    #loss += (torch.abs(std_x) - std_x) ** 2 + (torch.abs(std_y) - std_y) ** 2

    return torch.sum(loss * mask) / torch.sum(mask), mse_metric

def generator_step(args, batch, generator, optimizer_g):
    t_weight = 0.5

    obs_traj, obs_traj_rel, ground_truth_list, mask_list, render_list, seq_start_end = batch
    obs_traj = obs_traj.cuda()
    obs_traj_rel = obs_traj_rel.cuda()
    #ground_truth, mask, render, seq_start_end = ground_truth_list[0], mask_list[0], render_list[0], seq_start_end[0]
    outputs, times = generator(obs_traj, obs_traj_rel, seq_start_end)
    
    loss = torch.zeros(1).to(obs_traj)
    reg_weight = 1.0
    count = 0
    for i in range(len(outputs)):
        output = outputs[i]
        time = times[i]
        ground_truth, mask = ground_truth_list[i], mask_list[i]
        ground_truth = torch.tensor(ground_truth).cuda()
        mask = torch.tensor(mask).cuda()
        future_mask = mask > 0
        if (torch.sum(future_mask) == 0):
            continue
        count += 1

        output_loss, norm = loss_func_2d(output, ground_truth, future_mask)

        time_loss = loss_func_1d(time, mask, future_mask)

        tmp_loss = t_weight * time_loss + (1 - t_weight) * output_loss
        if loss is None:
            loss = tmp_loss
        else:
            loss += tmp_loss
    loss /= count

    optimizer_g.zero_grad()
    loss.backward()
    #for p in generator.parameters():
        #print(p.grad.norm())
    #print('\n')
    if args.clipping_threshold_g > 0:
        nn.utils.clip_grad_norm_(
            generator.parameters(), args.clipping_threshold_g
        )
    optimizer_g.step()
    return loss

def check_accuracy(args, loader, generator):
    
    generator.eval()
    total_loss = 0
    count = 0
    mse_metric = 0.0
    with torch.no_grad():
        for batch in loader: 
            obs_traj, obs_traj_rel, ground_truth_list, mask_list, render_list, seq_start_end = batch
            obs_traj = obs_traj.cuda()
            obs_traj_rel = obs_traj_rel.cuda()
            outputs, times = generator(obs_traj, obs_traj_rel, seq_start_end)
            
            loss_output = torch.zeros(1).to(obs_traj)
            for i in range(len(outputs)):
                output = outputs[i]
                time = times[i]
                ground_truth, mask = ground_truth_list[i], mask_list[i]
                ground_truth = torch.tensor(ground_truth).cuda()
                mask = torch.tensor(mask).cuda()
                future_mask = mask > 0
                if (torch.sum(future_mask) == 0):
                    continue
                count += 1

                output_loss, norm = loss_func_2d(output, ground_truth, future_mask)

                loss_output += output_loss
                mse_metric += norm
            total_loss += loss_output.item()
    generator.train()
    return total_loss / count, mse_metric / count


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)


#data loader, remove discriminator things, change loss function

#In model: modify pooling (n*n) x h to n x n x h
#          attach 1x1 convolution layers after pool_h in generator's forward function, output channel size is 2

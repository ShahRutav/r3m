# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from collections import namedtuple
from r3meval.utils.gym_env import GymEnv
from r3meval.utils.obs_wrappers import MuJoCoPixelObs, StateEmbedding
from r3meval.utils.sampling import sample_paths
from r3meval.utils.gaussian_mlp import MLP
from r3meval.utils.behavior_cloning import BC
from tabulate import tabulate
from tqdm import tqdm
import mj_envs, gym
import numpy as np, time as timer, multiprocessing, pickle, os
import os
import torch
from collections import namedtuple


#import metaworld
#from metaworld.envs import (ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE,
#                            ALL_V2_ENVIRONMENTS_GOAL_HIDDEN)

def set_bn_train(module):
    if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
        module.train()
    else:
        module.eval()

def set_model_train(model, last_n_layers=-1):
    if last_n_layers == -1: # Finetune all the layers
        model.train()
    elif last_n_layers == -2: # Finetune only BatchNorm layers
        model.apply(set_bn_train)
    elif last_n_layers > 0: # Finetune last n layers, define inside the model
        model.module.train_last_n_layers(last_n_layers)
    else:
        raise Exception

def tie_encs_(envs):
    #params = envs[0].env.embedding.parameters()
    #for e in envs:
    #    for p, new_p in zip(e.env.embedding.parameters(), params):
    #        p = new_p
    module = envs[0].env.embedding
    for e in envs:
        e.env.embedding = module

def check_layer(g, h, layer_no, layer_type):
    temp = ~torch.eq(g, h)
    if temp.any():
        print("Change in layerno {}, in layer type: {}".format(layer_no, layer_type))
    else:
        print("No Change in tensor")

def compare_params(mod1, mod2):
    for p1, p2 in zip(mod1.parameters(), mod2.parameters()):
        temp = ~torch.eq(p1, p2)
        if temp.any():
            print("Difference in params")
            raise Exception
        else:
            print("No difference")

def env_constructor(env_name, device='cuda', image_width=256, image_height=256,
                    camera_name=None, embedding_name='resnet50', pixel_based=True,
                    render_gpu_id=0, load_path="", proprio=False, lang_cond=False, gc=False):

    ## If pixel based will wrap in a pixel observation wrapper
    envs = []
    for env_ind, e_name in enumerate(env_name): # env_name is a list
        one_hot = np.zeros((len(env_name)))
        one_hot[env_ind] = 1
        one_hot = np.tile(one_hot, 1024)
        if pixel_based:
            ## Need to do some special environment config for the metaworld environments
            if "v2" in e_name:
                e  = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE[e_name]()
                e._freeze_rand_vec = False
                e.spec = namedtuple('spec', ['id', 'max_episode_steps'])
                e.spec.id = e_name
                e.spec.max_episode_steps = 500
            else:
                e = gym.make(e_name)
            ## Wrap in pixel observation wrapper
            e = MuJoCoPixelObs(e, width=image_width, height=image_height,
                               camera_name=camera_name, device_id=render_gpu_id)
            ## Wrapper which encodes state in pretrained model
            e = StateEmbedding(e, embedding_name=embedding_name, device=device, load_path=load_path,
                            proprio=proprio, camera_name=camera_name, env_name=e_name, env_embed=one_hot)
            e = GymEnv(e)
            envs.append(e)
        else:
            print("Only supports pixel based")
            assert(False)
    #compare_params(envs[0].env.embedding, envs[1].env.embedding)
    tie_encs_(envs) # Sets the embedding module same for all envs
    #compare_params(envs[0].env.embedding, envs[1].env.embedding)
    return envs

def get_bn_params(model):
    bn_paras = []
    for layer in model.modules():
       if isinstance(layer, torch.nn.modules.batchnorm.BatchNorm2d):
           bn_paras.extend(list(layer.parameters()))
    return bn_paras


def make_bc_agent(env_kwargs:dict, bc_kwargs:dict, demo_paths:list, epochs:int, seed:int, pixel_based=True):
    ## Creates environment
    envs = env_constructor(**env_kwargs)

    ## Creates MLP (Where the FC Network has a batchnorm in front of it)
    policy = MLP(envs[0].spec, hidden_sizes=(256, 256), seed=seed)
    policy.model.proprio_only = False

    ## Pass the encoder params to the BC agent (for finetuning)
    if pixel_based:
        if bc_kwargs['last_n_layers'] == -1:
            enc_p = envs[0].env.embedding.parameters()
        elif bc_kwargs['last_n_layers'] == -2:
            enc_p = get_bn_params(envs[0].env.embedding)
        elif bc_kwargs['last_n_layers'] > 0: # Finetune last n layers, define inside the model
            #enc_p = get_last_n_params(e.env.embedding.module.convnet)
            enc_p = envs[0].env.embedding.module.last_n_layerparams(bc_kwargs['last_n_layers'])
        else:
            raise Exception
    else:
        print("Only supports pixel based")
        assert(False)
    if bc_kwargs['pcgrad']:
        num_losses = len(env_kwargs['env_name'])
    else:
        num_losses = 1
    bc_agent = BC(demo_paths, policy=policy, epochs=epochs, set_transforms=False, encoder_params=enc_p, num_losses=num_losses, **bc_kwargs)

    ## Pass the environmetns observation encoder to the BC agent to encode demo data
    if pixel_based:
        bc_agent.encodefn = envs[0].env.encode_batch
    else:
        print("Only supports pixel based")
        assert(False)
    return envs, bc_agent


def configure_cluster_GPUs(gpu_logical_id: int) -> int:
    # get the correct GPU ID
    if "SLURM_STEP_GPUS" in os.environ.keys():
        physical_gpu_ids = os.environ.get('SLURM_STEP_GPUS')
        gpu_id = int(physical_gpu_ids.split(',')[gpu_logical_id])
        print("Found slurm-GPUS: <Physical_id:{}>".format(physical_gpu_ids))
        print("Using GPU <Physical_id:{}, Logical_id:{}>".format(gpu_id, gpu_logical_id))
    else:
        gpu_id = 0 # base case when no GPUs detected in SLURM
        print("No GPUs detected. Defaulting to 0 as the device ID")
    return gpu_id

def add_one_hot_enc(demo_paths, ind, tot_len):
    for demo_ind in range(len(demo_paths)):
        one_hot = np.zeros((demo_paths[demo_ind]["observations"].shape[0], tot_len))
        one_hot[:, ind] = 1
        one_hot = np.tile(one_hot, (1, 1024) )
        demo_paths[demo_ind]["observations"] = np.concatenate([one_hot, demo_paths[demo_ind]["observations"]], -1)
    return demo_paths

def bc_train_loop(job_data:dict) -> None:

    # configure GPUs
    os.environ['GPUS'] = os.environ.get('SLURM_STEP_GPUS', '0')
    physical_gpu_id = 0 #configure_cluster_GPUs(job_data['env_kwargs']['render_gpu_id'])
    job_data['env_kwargs']['render_gpu_id'] = physical_gpu_id

    if len(job_data['env_kwargs']['env_name']) > 1:
        job_data['proprio'] += 1024*len(job_data['env_kwargs']['env_name'])

    # Infers the location of the demos
    ## V2 is metaworld, V0 adroit, V3 kitchen
    #data_dir = '/iris/u/surajn/data/r3m/'
    data_dir = '/home/bt1/18CS10050/r3m/r3m-data/'
    demo_paths = None
    for env_ind, env_name in enumerate(job_data['env_kwargs']['env_name']):
        print("Loading demos for", env_name)
        if "v2" in env_name:
            demo_paths_loc = data_dir + 'final_paths_multiview_meta_200/' + job_data['camera'] + '/' + env_name + '.pickle'
        elif "v0" in env_name:
            demo_paths_loc = data_dir + 'final_paths_multiview_adroit_200/' + job_data['camera'] + '/' + env_name + '.pickle'
        else:
            demo_paths_loc = data_dir + 'final_paths_multiview_rb_200/' + job_data['camera'] + '/' + env_name + '.pickle'

        ## Loads the demos
        if demo_paths is None:
            demo_paths = pickle.load(open(demo_paths_loc, 'rb'))
            demo_paths = demo_paths[:job_data['num_demos']]
            demo_paths = add_one_hot_enc(demo_paths, env_ind, len(job_data['env_kwargs']['env_name']))
        else:
            new_demos = pickle.load(open(demo_paths_loc, 'rb'))
            new_demos = new_demos[:job_data['num_demos']]
            new_demos = add_one_hot_enc(new_demos, env_ind, len(job_data['env_kwargs']['env_name']))
            for p in new_demos:
                demo_paths.append(p)

        print(len(demo_paths))
    demo_score = np.mean([np.sum(p['rewards']) for p in demo_paths])
    print("Demonstration score : %.2f " % demo_score)

    # Make log dir
    if os.path.isdir(job_data['job_name']) == False: os.mkdir(job_data['job_name'])
    previous_dir = os.getcwd()
    os.chdir(job_data['job_name']) # important! we are now in the directory to save data
    if os.path.isdir('iterations') == False: os.mkdir('iterations')
    if os.path.isdir('logs') == False: os.mkdir('logs')

    ## Creates agent and environment
    env_kwargs = job_data['env_kwargs']
    envs, agent = make_bc_agent(env_kwargs=env_kwargs, bc_kwargs=job_data['bc_kwargs'],
                             demo_paths=demo_paths, epochs=1, seed=job_data['seed'], pixel_based=job_data["pixel_based"])
    agent.logger.init_wb(job_data)

    highest_score = -np.inf
    max_success = 0
    epoch = 0
    #print(e.env.embedding.module.convnet)
    ten_0_c, ten_0_b, ten_1_c, ten_1_b, ten_2_c, ten_2_b, ten_3_c, ten_3_b = None, None, None, None, None, None, None, None
    ten_0_a, ten_1_a, ten_2_a, ten_3_a = None, None, None, None
    while True:
        # update policy using one BC epoch
        last_step = agent.steps
        print("Step", last_step)
        #compare_params(envs[0].env.embedding, envs[1].env.embedding)
        agent.policy.model.train()
        # If finetuning, wait until 25% of training is done then
        ## set embedding to train mode and turn on finetuning
        if (job_data['bc_kwargs']['finetune']) and (job_data['pixel_based']) and (job_data['env_kwargs']['load_path'] != "clip"):
            if last_step > (job_data['steps'] / 4.0):
                if False: # Clean this part. only for debugging finetune
                    if ten_0_b is None:
                        ten_0_b = envs[0].env.embedding.module.convnet.layer1[0].bn3.weight.clone()
                        ten_0_c = envs[0].env.embedding.module.convnet.layer1[0].conv3.weight.clone()
                        ten_0_a = envs[0].env.embedding.module.convnet.layer1[0].bn3.running_mean.clone()
                        ten_1_b = envs[0].env.embedding.module.convnet.layer2[0].bn3.weight.clone()
                        ten_1_c = envs[0].env.embedding.module.convnet.layer2[0].conv3.weight.clone()
                        ten_1_a = envs[0].env.embedding.module.convnet.layer2[0].bn3.running_mean.clone()
                        ten_2_b = envs[0].env.embedding.module.convnet.layer3[0].bn3.weight.clone()
                        ten_2_c = envs[0].env.embedding.module.convnet.layer3[0].conv3.weight.clone()
                        ten_2_a = envs[0].env.embedding.module.convnet.layer3[0].bn3.running_mean.clone()
                        ten_3_b = envs[0].env.embedding.module.convnet.layer4[0].bn3.weight.clone()
                        ten_3_c = envs[0].env.embedding.module.convnet.layer4[0].conv3.weight.clone()
                        ten_3_a = envs[0].env.embedding.module.convnet.layer4[0].bn3.running_mean.clone()
                    else:
                        check_layer(ten_0_b, envs[0].env.embedding.module.convnet.layer1[0].bn3.weight, 1, 'BN')
                        check_layer(ten_0_c, envs[0].env.embedding.module.convnet.layer1[0].conv3.weight, 1, 'conv')
                        check_layer(ten_0_a, envs[0].env.embedding.module.convnet.layer1[0].bn3.running_mean, 1, 'running_mean')
                        check_layer(ten_1_b, envs[0].env.embedding.module.convnet.layer2[0].bn3.weight, 2, 'BN')
                        check_layer(ten_1_c, envs[0].env.embedding.module.convnet.layer2[0].conv3.weight, 2, 'conv')
                        check_layer(ten_1_a, envs[0].env.embedding.module.convnet.layer2[0].bn3.running_mean, 2, 'running_mean')
                        check_layer(ten_2_b, envs[0].env.embedding.module.convnet.layer3[0].bn3.weight, 3, 'BN')
                        check_layer(ten_2_c, envs[0].env.embedding.module.convnet.layer3[0].conv3.weight, 3, 'conv')
                        check_layer(ten_2_a, envs[0].env.embedding.module.convnet.layer3[0].bn3.running_mean, 3, 'running_mean')
                        check_layer(ten_3_b, envs[0].env.embedding.module.convnet.layer4[0].bn3.weight, 4, 'BN')
                        check_layer(ten_3_c, envs[0].env.embedding.module.convnet.layer4[0].conv3.weight, 4, 'conv')
                        check_layer(ten_3_a, envs[0].env.embedding.module.convnet.layer4[0].bn3.running_mean, 4, 'running_mean')
                    #print(temp)
                set_model_train(envs[0].env.embedding, last_n_layers=job_data['bc_kwargs']['last_n_layers'])
                #e.env.embedding.train()
                envs[0].env.set_finetuning(True)
        agent.train(job_data['pixel_based'], suppress_fit_tqdm=True, step = last_step)

        # perform evaluation rollouts every few epochs
        if ((agent.steps % job_data['eval_frequency']) < (last_step % job_data['eval_frequency'])):
            agent.policy.model.eval()
            agent.logger.log_kv('eval_epoch', epoch)
            avg_success = 0.0
            for env_name, e in zip(job_data['env_kwargs']['env_name'], envs):
                if job_data['pixel_based']:
                    e.env.embedding.eval()
                paths = sample_paths(num_traj=job_data['eval_num_traj'], env=e, #env_constructor,
                                     policy=agent.policy, eval_mode=True, horizon=e.horizon,
                                     base_seed=job_data['seed']+epoch, num_cpu=job_data['num_cpu'],
                                     env_kwargs=env_kwargs)

                try:
                    ## Success computation and logging for Adroit and Kitchen
                    success_percentage = e.env.unwrapped.evaluate_success(paths)
                    for i, path in enumerate(paths):
                        if (i < 10) and job_data['pixel_based']:
                            vid = path['images']
                            filename = f'./iterations/vid_{env_name}_{i}.gif'
                            from moviepy.editor import ImageSequenceClip
                            cl = ImageSequenceClip(vid, fps=20)
                            cl.write_gif(filename, fps=20)
                except:
                    ## Success computation and logging for MetaWorld
                    sc = []
                    for i, path in enumerate(paths):
                        sc.append(path['env_infos']['success'][-1])
                        if (i < 10) and job_data['pixel_based']:
                            vid = path['images']
                            filename = f'./iterations/vid_{env_name}_{i}.gif'
                            from moviepy.editor import ImageSequenceClip
                            cl = ImageSequenceClip(vid, fps=20)
                            cl.write_gif(filename, fps=20)
                    success_percentage = np.mean(sc) * 100

                avg_success += success_percentage
                agent.logger.log_kv('eval_success_{}'.format(env_name), success_percentage)

                # Tracking best success over training
                max_success = max(max_success, success_percentage)

            avg_success /= len(envs)
            agent.logger.log_kv('avg_eval_success', avg_success)
            # save policy and logging
            pickle.dump(agent.policy, open('./iterations/policy_%i.pickle' % epoch, 'wb'))
            agent.logger.save_log('./logs/')
            agent.logger.save_wb(step=agent.steps)

            print_data = sorted(filter(lambda v: np.asarray(v[1]).size == 1,
                                        agent.logger.get_current_log().items()))
            print(tabulate(print_data))
        epoch += 1
        if agent.steps > job_data['steps']:
            break
    agent.logger.log_kv('max_success', max_success)
    agent.logger.save_wb(step=agent.steps)


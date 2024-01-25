from agents import *
from envs import *
from utils import *
from config import *
from torch.multiprocessing import Pipe


import numpy as np
import pickle
from utils import Logger, set_seed
from os import path

from dist_utils import ddp_setup, create_parallel_env_processes
import torch.distributed as dist

def main(args):
    logger = Logger(file_log_path=path.join("logs", "file_logs", args['log_name']), tb_log_path=path.join("logs", "tb_logs", args['log_name']))
    use_cuda = default_config.getboolean('UseGPU')

    GLOBAL_WORLD_SIZE, GLOBAL_RANK, LOCAL_WORLD_SIZE, LOCAL_RANK, gpu_id = ddp_setup(logger, use_cuda)

    dist.barrier() # wait for process initialization logging inside ddp_setup() to finish
    assert GLOBAL_WORLD_SIZE == 1 and LOCAL_WORLD_SIZE == 1, "There should only be only 1 process !"


    logger.log_msg_to_both_console_and_file(
        "*" * 30 + "\n" +
        str(dict(**{section: dict(config[section]) for section in config.sections()}, **args)) + "\n"
        + "\n" + "*" * 30,
        only_rank_0=True
        )

    num_env_workers = int(args['num_env_per_process'])
    assert num_env_workers == 1, "num_env_per_process has to be 1"

    seed = args['seed'] + (GLOBAL_RANK * num_env_workers) # set different seed to every env_worker process so that every env does not play the same game
    set_seed(args['seed']) # Note: this will not seed the gym environment

    train_method = default_config['TrainMethod']
    assert train_method in ['PPO', 'original_RND', 'modified_RND']
    representation_lr_method = str(default_config['representationLearningMethod'])

    env_id = default_config['EnvID']
    env_type = default_config['EnvType']

    if env_type == 'mario':
        from nes_py.wrappers import JoypadSpace
        # env = BinarySpaceToDiscreteSpaceEnv(gym_super_mario_bros.make(env_id), COMPLEX_MOVEMENT)
        env = JoypadSpace(gym_super_mario_bros.make(env_id, apply_api_compatibility=True), COMPLEX_MOVEMENT)
    elif env_type == 'atari':
        env = gym.make(env_id)
    elif env_type == 'classic_control':
        env = gym.make(env_id)
    else:
        raise NotImplementedError

    if default_config['PreProcHeight'] is not None:
        input_size = int(default_config['PreProcHeight'])
    else:
        input_size = env.observation_space.shape  # 4

    if isinstance(env.action_space, gym.spaces.box.Box):
        output_size = env.action_space.shape[0]
    else:
        output_size = env.action_space.n  # 2

    if 'Breakout' in env_id:
        output_size -= 1

    env.close()

    is_load_model = True
    is_render = True
    load_ckpt_path = '{}'.format(args['load_model_path']) # path for resuming a training from a checkpoint

    use_gae = default_config.getboolean('UseGAE')
    use_noisy_net = default_config.getboolean('UseNoisyNet')

    GAE_Lambda = float(default_config['GAELambda'])

    num_step = int(default_config['NumStep'])

    ppo_eps = float(default_config['PPOEps'])
    epoch = int(default_config['Epoch'])
    mini_batch = int(default_config['MiniBatch'])
    batch_size = int(num_step * num_env_workers / mini_batch)
    learning_rate = float(default_config['LearningRate'])
    entropy_coef = float(default_config['Entropy'])
    gamma = float(default_config['Gamma'])
    max_grad_norm = float(default_config['MaxGradNorm'])

    sticky_action = False
    stateStackSize = int(default_config['StateStackSize'])
    action_prob = float(default_config['ActionProb'])
    life_done = default_config.getboolean('LifeDone')

    agent = RNDAgent

    if default_config['EnvType'] == 'atari':
        env_type = AtariEnvironment
    elif default_config['EnvType'] == 'mario':
        env_type = MarioEnvironment
    elif default_config['EnvType'] == 'classic_control':
        env_type = ClassicControlEnvironment
    else:
        raise NotImplementedError
    

    # Create environment processes
    _, env_worker_parent_conns, _ = create_parallel_env_processes(num_env_workers, env_type, env_id, sticky_action, action_prob, life_done, stateStackSize, input_size, seed, logger)

    agent = agent(
        input_size,
        output_size,
        num_env_workers,
        num_step,
        gamma,
        GAE_Lambda=GAE_Lambda,
        learning_rate=learning_rate,
        ent_coef=entropy_coef,
        max_grad_norm=max_grad_norm,
        epoch=epoch,
        batch_size=batch_size,
        ppo_eps=ppo_eps,
        use_cuda=use_cuda,
        use_gae=use_gae,
        use_noisy_net=use_noisy_net,
        representation_lr_method=representation_lr_method,
        device=gpu_id,
        logger=logger
    )

    global_step = 0
    undiscounted_episode_return = deque([], maxlen=100)
    episode_lengths = deque([], maxlen=100)
    if 'Montezuma' in env_id:
        number_of_visited_rooms = deque([], maxlen=100)

    if is_load_model:
        logger.log_msg_to_both_console_and_file(f'loading from checkpoint: {load_ckpt_path}', only_rank_0=True)
        if use_cuda:
            load_checkpoint = torch.load(load_ckpt_path, map_location=gpu_id)
            agent.load_state_dict(load_checkpoint['agent.state_dict'])
            # agent.model.load_state_dict(load_checkpoint['agent.model.state_dict'])
            # agent.rnd.predictor.load_state_dict(load_checkpoint['agent.rnd.predictor.state_dict'])
            # agent.rnd.target.load_state_dict(load_checkpoint['agent.rnd.target.state_dict'])
            if representation_lr_method == "BYOL": # BYOL
                # agent.representation_model.load_state_dict(load_checkpoint['agent.representation_model.state_dict'])
                assert agent.representation_model.net is agent.model.feature
                # agent.representation_model.net = agent.model.feature # representation_model's net should map to the feature extractor of the RL algo
            if representation_lr_method == "Barlow-Twins": # Barlow-Twins
                # agent.representation_model.load_state_dict(load_checkpoint['agent.representation_model.state_dict'])
                # agent.representation_model.backbone = agent.model.feature # representation_model's net should map to the feature extractor of the RL algo
                assert agent.representation_model.backbone is agent.model.feature

        else:
            load_checkpoint = torch.load(load_ckpt_path, map_location='cpu')
            agent.load_state_dict(load_checkpoint['agent.state_dict'])
            # agent.model.load_state_dict(load_checkpoint['agent.model.state_dict'])
            # agent.rnd.predictor.load_state_dict(load_checkpoint['agent.rnd.predictor.state_dict'])
            # agent.rnd.target.load_state_dict(load_checkpoint['agent.rnd.target.state_dict'])
            if representation_lr_method == "BYOL": # BYOL
                # agent.representation_model.load_state_dict(load_checkpoint['agent.representation_model.state_dict'])
                # agent.representation_model.net = agent.model.feature # representation_model's net should map to the feature extractor of the RL algo
                assert agent.representation_model.net is agent.model.feature
            if representation_lr_method == "Barlow-Twins": # Barlow-Twins
                # agent.representation_model.load_state_dict(load_checkpoint['agent.representation_model.state_dict'])
                # agent.representation_model.backbone = agent.model.feature # representation_model's net should map to the feature extractor of the RL algo
                assert agent.representation_model.backbone is agent.model.feature


        if train_method in ['original_RND', 'modified_RND']:
            for param in agent.rnd.target.parameters():
                assert param.requires_grad == False

        obs_rms = load_checkpoint['obs_rms']
        reward_rms = load_checkpoint['reward_rms']

        logger.log_msg_to_both_console_and_file('loading finished!', only_rank_0=True)


    agent_PPO_total_params = sum(p.numel() for p in agent.model.parameters())
    agent_RND_predictor_total_params = sum(p.numel() for p in agent.rnd.predictor.parameters()) if agent.rnd is not None else 0
    agent_representation_model_total_params = sum(p.numel() for p in agent.representation_model.parameters()) if agent.representation_model is not None else 0
    logger.log_msg_to_both_console_and_file(f"{'*'*20}\
        \nNumber of PPO parameters: {agent_PPO_total_params}\
        \nNumber of RND_predictor parameters: {agent_RND_predictor_total_params}\
        \nNumber of {representation_lr_method if representation_lr_method != 'None' else 'Representation Learning Model'} parameters: {agent_representation_model_total_params}\
        \n{'*'*20}", only_rank_0=True)

    agent.set_mode("eval")


    states = np.zeros([num_env_workers, stateStackSize, input_size, input_size]) # [num_env_workers, stateStackSize, input_size, input_size]

    if is_render:
        renderer = ParallelizedEnvironmentRenderer(num_env_workers)
    while True:
        total_state, total_reward, total_done, total_action, total_int_reward, total_next_obs, total_ext_values, total_int_values, total_policy = \
            [], [], [], [], [], [], [], [], []
        global_step += 1
        with torch.no_grad():
            actions, value_ext, value_int, policy = agent.get_action(np.float32(states) / 255.)
            actions = torch.tensor(actions, dtype=torch.int64)

        for parent_conn, action in zip(env_worker_parent_conns, actions):
                parent_conn.send(action)

        next_states, rewards, dones, next_obs = [], [], [], []
        for env_idx, parent_conn in enumerate(env_worker_parent_conns):
            s, r, d, trun, visited_rooms = parent_conn.recv()
            assert (list(s.shape) == [stateStackSize, input_size, input_size]) and (s.dtype == np.float64)
            assert type(r) == float
            assert type(d) == bool
            assert type(trun) == bool

            next_states.append(s)
            rewards.append(r)
            dones.append(d) # --> [num_env]
            if train_method == 'original_RND':
                next_obs.append(s[(stateStackSize - 1), :, :].reshape([1, input_size, input_size])) # [1, input_size, input_size]
            elif train_method == 'modified_RND':
                next_obs.append(s) # [stateStackSize, input_size, input_size]


            if d or trun:
                info = {'episode': {}}
                if 'Montezuma' in env_id:
                    info['episode']['number_of_visited_rooms'], info['episode']['visited_rooms'] = parent_conn.recv()
                    number_of_visited_rooms.append(info['episode']['number_of_visited_rooms'])
                info['episode']['undiscounted_episode_return'], info['episode']['l'], info['episode']['num_finished_episodes'] = parent_conn.recv()
                undiscounted_episode_return.append(info['episode']['undiscounted_episode_return'])
                episode_lengths.append(info['episode']['l'])
                # Logging:
                if 'Montezuma' in env_id:
                    logger.log_msg_to_console(f'[Rank: {GLOBAL_RANK}, env: {env_idx}] episode: {info["episode"]["num_finished_episodes"]}, step: {info["episode"]["l"]}, undiscounted_return: {info["episode"]["undiscounted_episode_return"]}, moving_average_undiscounted_return: {np.mean(info["episode"]["undiscounted_episode_return"])}, visited_rooms: {info["episode"]["visited_rooms"]}')
                else:
                    logger.log_msg_to_console(f'[Rank: {GLOBAL_RANK}, env: {env_idx} ] episode: {info["episode"]["num_finished_episodes"]}, step: {info["episode"]["l"]}, undiscounted_return: {info["episode"]["undiscounted_episode_return"]}, moving_average_undiscounted_return: {np.mean(info["episode"]["undiscounted_episode_return"])}')
        

        next_states = np.stack(next_states) # -> [num_env, state_stack_size, H, W]
        rewards = np.hstack(rewards) # -> [num_env, ]
        dones = np.hstack(dones) # -> [num_env, ]
        assert (list(next_states.shape) == [num_env_workers, stateStackSize, input_size, input_size]) and (next_states.dtype == np.float64)
        assert (list(rewards.shape) == [num_env_workers, ]) and (rewards.dtype == np.float64)
        assert (list(dones.shape) == [num_env_workers, ]) and (dones.dtype == np.bool_)
        if train_method in ['original_RND', 'modified_RND']:
            next_obs = np.stack(next_obs) # -> modified_RND: [num_env, stateStackSize, H, W], original_RND: [num_env, 1, H, W]


        # Compute normalize obs, compute intrinsic rewards and clip them (note that: total reward = int reward + ext reward)
        if train_method == 'original_RND':
            intrinsic_reward = agent.compute_intrinsic_reward(
                ((next_obs - obs_rms.mean) / np.sqrt(obs_rms.var)).clip(-5, 5)) # -> [num_env, ]
                # ((next_obs.cpu().numpy() - obs_rms.mean) / np.sqrt(obs_rms.var)).clip(-5, 5)) # -> [num_env, ]
        elif train_method == 'modified_RND':
            with torch.no_grad(): # gradients should not flow backwards from RND to the PPO's bacbone (i.e. RND gradients should stop at the feature embeddings extracted by the PPO's bacbone)
                extracted_feature_embeddings = agent.extract_feature_embeddings(next_obs / 255).cpu().numpy() # [num_worker_envs=1, feature_embeddings_dim]
                intrinsic_reward = agent.compute_intrinsic_reward(
                    ((extracted_feature_embeddings - obs_rms.mean) / np.sqrt(obs_rms.var)).clip(-5, 5)) # -> [num_env, ]
        

        if train_method in ['original_RND', 'modified_RND']:
            intrinsic_reward = np.hstack(intrinsic_reward) # [num_env,]

        total_next_obs.append(next_obs) # --> modified_RND: [num_step, num_env, state_stack_size, H, W], original_RND: [num_step, num_env, 1, H, W]
        if train_method in ['original_RND', 'modified_RND']:
            total_int_reward.append(intrinsic_reward)
        total_state.append(states)
        total_reward.append(rewards)
        total_done.append(dones)
        total_action.append(actions)
        total_ext_values.append(value_ext)
        total_int_values.append(value_int)
        total_policy.append(policy)

        # normalize intrinsic reward
        if train_method in ['original_RND', 'modified_RND']:
            # running mean intrinsic reward
            total_int_reward = np.stack(total_int_reward).transpose() # --> [num_env, num_step]
            # normalize intrinsic reward
            total_int_reward /= np.sqrt(reward_rms.var)

        states = next_states[:, :, :, :] # for an explanation of why [:, :, :, :] is used refer to the discussion: https://stackoverflow.com/questions/61103275/what-is-the-difference-between-tensor-and-tensor-in-pytorch 
                
        # if done:
        #     intrinsic_reward_list = (intrinsic_reward_list - np.mean(intrinsic_reward_list)) / np.std(
        #         intrinsic_reward_list)
        #     with open('int_reward', 'wb') as f:
        #         pickle.dump(intrinsic_reward_list, f)


        if is_render:
            if train_method == 'original_RND':
                renderer.render(next_obs) # [num_env, 1, input_size, input_size]
            elif train_method == 'modified_RND':
                renderer.render(next_obs[:, -1, :, :].reshape([num_env_workers, 1, input_size, input_size])) # [num_env, 1, input_size, input_size]
            elif train_method == 'PPO':
                renderer.render(next_states[:, -1, :, :].reshape([num_env_workers, 1, input_size, input_size])) # [num_env, 1, input_size, input_size]

    if is_render:
        renderer.close()

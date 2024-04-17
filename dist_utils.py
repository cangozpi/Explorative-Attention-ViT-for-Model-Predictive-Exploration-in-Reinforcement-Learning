import os
import torch
from torch.distributed import init_process_group, destroy_process_group
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.multiprocessing import Pipe
from config import *

def get_dist_info():
    GLOBAL_WORLD_SIZE = int(os.environ["WORLD_SIZE"])
    GLOBAL_RANK = int(os.environ["RANK"])
    LOCAL_WORLD_SIZE = int(os.environ["LOCAL_WORLD_SIZE"])
    LOCAL_RANK = int(os.environ["LOCAL_RANK"])
    return GLOBAL_WORLD_SIZE, GLOBAL_RANK, LOCAL_WORLD_SIZE, LOCAL_RANK


def ddp_setup(logger, use_cuda, gpu_id):
    """
    Setups torch.distributed and creates p_groups for training.
    In every node, the processes created by torch distributed (i.e. processes belonging to the default process group), will act 
    as the agent_processes which contain DPP covered Deep RL agent and perform training. Each of these agent processes will get a unique
    GPU, so you should have as many GPUs available as the number of agent processes you create in each node. Also, each agent process will
    manually spawn --num_env_per_process many environment processes which it will communicate over python multiprocessing Pipe to interact with the 
    environments in parallel. To get a better understanding check out the example below:
    In every node, 1 process (process with local_rank == 0) is assigned to the agents_group, the remaining processes are
    assigned to the env_workers_group. To get a better understanding check out the example below.


    Example:
        Available from torchrun:
            nnodes: number of nodes = 3
            nproc_per_node: number of processes per node = 2
            num_env_per_agent_process = 16
        ---

        ************** NODE 0:
        LOCAL_RANK 0: GPU0 --> agent process 0: spawns 16 environment processes (env processes 0 ... 15) and interact with them
        LOCAL_RANK 1: GPU1 --> agent process 1: spawns 16 environment processes (env processes 16 ... 31) and interact with them
        **************
        ...

        ************** NODE: 1:
        LOCAL_RANK 2: GPU2 --> agent process 2: spawns 16 environment processes (env processes 32 ... 47) and interact with them
        LOCAL_RANK 3: GPU3 --> agent process 3: spawns 16 environment processes (env processes 48 ... 63) and interact with them
        **************
        ...

        ************** NODE: 2:
        LOCAL_RANK 4: GPU4 --> agent process 4: spawns 16 environment processes (env processes 64 ... 79) and interact with them
        LOCAL_RANK 5: GPU5 --> agent process 5: spawns 16 environment processes (env processes 80 ... 95) and interact with them
        **************

        -node0-  -node1-   -node2-
          0,1,     2,3,      4,5    ||
        *        *        *

    """

    GLOBAL_WORLD_SIZE, GLOBAL_RANK, LOCAL_WORLD_SIZE, LOCAL_RANK = get_dist_info()

    if use_cuda:
        assert torch.cuda.is_available() == True, "use_cuda:True is passed but cuda is not available !"

    if torch.cuda.is_available() and use_cuda and (gpu_id == -1):
        gpu_id = "cuda:" + str(LOCAL_RANK % torch.cuda.device_count())
        backend = "nccl"
    elif torch.cuda.is_available() and use_cuda and (gpu_id != -1):
        gpu_id = "cuda:" + str(gpu_id)
        backend = "nccl"
        assert GLOBAL_WORLD_SIZE == 1, "when --gpu_id is passed then the GLOBAL_WORLD_SIZE must be 1"
    else:
        gpu_id = "cpu"
        backend = "gloo"

    if torch.cuda.is_available() and use_cuda:
        torch.cuda.set_device(gpu_id) # each process should have a unique cuda device (otherwise see the error at: https://github.com/pytorch/torchrec/issues/328)
    init_process_group(backend=backend)

    logger.log_msg_to_both_console_and_file(f'Initialized process with global_rank: {GLOBAL_RANK}, local_rank: {LOCAL_RANK}, local_world_size: {LOCAL_WORLD_SIZE}, global_world_size: {GLOBAL_WORLD_SIZE}, [{gpu_id}], backend: {backend}')
    
    return GLOBAL_WORLD_SIZE, GLOBAL_RANK, LOCAL_WORLD_SIZE, LOCAL_RANK, gpu_id 


def create_parallel_env_processes(num_env_per_process, env_type, env_id, sticky_action, action_prob, life_done, stateStackSize, input_size, seed, logger):
    """
    Creates num_env_per_process many environment processes. Each one of these processes run a gym environment.
    Returns:
        env_workers (list): contains references to started environment processes.
        parent_conns (list): contains one end of the Pipe connnections which are used for IPC btw main process and the environment processes.
        child_conns (list): contains the other end of the Pipe connnections which are used for IPC btw main process and the environment processes.
    """
    GLOBAL_WORLD_SIZE, GLOBAL_RANK, LOCAL_WORLD_SIZE, LOCAL_RANK = get_dist_info()

    if default_config['EnvType'] == 'atari': # other env_type's raise a pickling related error
        # mp.set_start_method('spawn', force=True) # required to avoid python's GIL issue, (also logger cannot be passed to env process constructor, see the issue: https://discuss.pytorch.org/t/thread-lock-object-cannot-be-pickled-when-using-pytorch-multiprocessing-package-with-spawn-method/184953)
        try:
            mp.set_start_method('spawn')
        except RuntimeError:
            pass

    env_workers = []
    parent_conns = []
    child_conns = []
    for idx in range(num_env_per_process):
        parent_conn, child_conn = Pipe()
        
        from copy import deepcopy
        env_worker = env_type(env_id=env_id, is_render=False, env_idx=(GLOBAL_RANK*num_env_per_process)+idx, sticky_action=sticky_action, p=action_prob, h=input_size, w=input_size,
                            life_done=life_done, history_size=stateStackSize, seed=seed+((GLOBAL_RANK*num_env_per_process)+idx), child_conn=child_conn) # Note that seed+rank is required to make parallel envs play different scenarios
        env_worker.start()
        env_workers.append(env_worker)
        parent_conns.append(parent_conn)
        child_conns.append(child_conn)

        logger.log_msg_to_both_console_and_file(f'Created environment process worker_env: {idx}, for agent process [rank={GLOBAL_RANK}]')
    
    return env_workers, parent_conns, child_conns


def distributed_cleanup():
    dist.destroy_process_group()
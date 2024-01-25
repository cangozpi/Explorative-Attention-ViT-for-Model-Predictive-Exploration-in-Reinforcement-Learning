docker_build:
	docker build --no-cache . -t rl_image

# docker_start:
# 	docker run --name rl -e DISPLAY={DISPLAY} --net=host -v ~/Desktop/Docker_shared/Representation\ Learning\ for\ Model-Predictive\ Exploration\ in\ Reinforcement\ Learning:/Docker_shared/Representation\ Learning\ for\ Model-Predictive\ Exploration\ in\ Reinforcement\ Learning \
# 	 -it --privileged rl_image

docker_start:
	docker run --name rl -it --gpus all --net host rl_image

docker_rm:
	docker container rm rl && \
	docker image rm rl_image && \
	docker system prune -a

train:
	# torchrun --nnodes 1 --nproc_per_node 2 --standalone main.py --train --num_env_per_process 3 --config_path=./configs/demo_config.conf --log_name=demo_00 --save_model_path=checkpoints/demo_00.ckpt 
	torchrun --nnodes 1 --nproc_per_node 1 --standalone main.py --train --num_env_per_process 3 --config_path=./configs/demo_config.conf --log_name=demo_00 --save_model_path=checkpoints/demo_00.ckpt 

continue_training:
	torchrun --nnodes 1 --nproc_per_node 3 --standalone main.py --train --config_path=./configs/demo_config.conf --log_name=demo_00_cont00 --load_model_path=checkpoints/demo_00.ckpt --save_model_path=checkpoints/demo_00_cont00.ckpt 
	# torchrun --nnodes 1 --nproc_per_node 3 --standalone main.py --train --config_path=./configs/demo_config.conf --log_name=demo_00_cont00 --load_model_path=checkpoints/demo_00__BestModelForMeanExtrinsicRolloutRewards.ckpt --save_model_path=checkpoints/demo_00_cont00.ckpt 
	# torchrun --nnodes 1 --nproc_per_node 3 --standalone main.py --train --config_path=./configs/demo_config.conf --log_name=demo_00_cont00 --load_model_path=checkpoints/demo_00__BestModelForMeanUndiscountedEpisodeReturn.ckpt --save_model_path=checkpoints/demo_00_cont00.ckpt 

pytorch_profiling:
	torchrun --nnodes 1 --nproc_per_node 3 --standalone main.py --train --config_path=./configs/demo_config.conf --log_name=demo_00 --save_model_path=checkpoints/demo_00.ckpt --pytorch_profiling

scalene_profiling:
	# python -m scalene --- -m torch.distributed.run --nnodes 1 --nproc_per_node 3 --standalone main.py --train --config_path=./configs/demo_config.conf --log_name=demo_00 --save_model_path=checkpoints/demo_00.ckpt --scalene_profiling 3
	# python -m scalene --no-browser --cpu --gpu --memory --outfile scaleneProfiler00_rnd00.html --profile-interval 10 --- -m torch.distributed.run --nnodes 1 --nproc_per_node 3 --standalone main.py --train --config_path=./configs/demo_config.conf --log_name=demo_00 --save_model_path=checkpoints/demo_00.ckpt --scalene_profiling 3
	python -m scalene --no-browser --outfile scaleneProfiler00_rnd00.html --- -m torch.distributed.run --nnodes 1 --nproc_per_node 3 --standalone main.py --train --config_path=./configs/demo_config.conf --log_name=demo_00 --save_model_path=checkpoints/demo_00.ckpt --scalene_profiling 3

test:
	torchrun --nnodes 1 --nproc_per_node 1 --standalone main.py --eval --num_env_per_process 1 --config_path=./configs/demo_config.conf --log_name=demo_test_00 --load_model_path=checkpoints/demo_00.ckpt

kill:
	kill -9 $(shell pidof python);
	kill -9 $(shell pidof python3)

start_tensorboard:
	tensorboard --logdir logs/tb_logs

start_tensorboard_profiles:
	tensorboard --logdir=./logs/torch_profiler_logs

run_tests:
	python3 test.py --train

scalene_profiling:
	python -m scalene --- -m torch.distributed.run --nnodes 1 --nproc_per_node 3 --standalone main.py --train --config_path=./configs/demo_config.conf --log_name=demo_00 --save_model_path=checkpoints/demo_00.ckpt 

# test:
# 	torchrun --nnodes 1 --nproc_per_node 2 --standalone main.py --eval --config_path=./configs/demo_config.conf --log_name=demo_test_00 --load_model_path=checkpoints/demo_00.ckpt

kill:
	kill -9 $(shell pidof python);
	kill -9 $(shell pidof python3)

start_tensorboard:
	tensorboard --logdir logs/tb_logs

start_tensorboard_profiles:
	tensorboard --logdir=./logs/torch_profiler_logs

run_tests:
	python3 test.py --train
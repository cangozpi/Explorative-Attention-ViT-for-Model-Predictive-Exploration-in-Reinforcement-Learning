from config import *
from utils import print_config_options

if __name__ == '__main__':
    if args['config_options']:
        print_config_options()
    if args['train']:
        from train import main
    elif args['eval']:
        from eval import main
    main(args)
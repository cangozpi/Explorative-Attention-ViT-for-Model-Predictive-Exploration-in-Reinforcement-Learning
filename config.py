import configparser
from argument_parser import get_args

# helper
def set_debug_params():
    """
    Sets some hyperparameters to small values to debug more easily.
    """
    args['num_env_per_process'] = 4
    config.set(default, 'NumStep', '16')
    config.set(default, 'ObsNormStep', '5')

# 
args = get_args()

default = 'DEFAULT'

config = configparser.ConfigParser()
config.read(args['config_path'])
if args['debug_params']:
    set_debug_params()

# ---------------------------------
# ---------------------------------
default_config = config[default]

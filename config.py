import configparser
from argument_parser import get_args

args = get_args()

config = configparser.ConfigParser()
config.read(args['config_path'])

# ---------------------------------
default = 'DEFAULT'
# ---------------------------------
default_config = config[default]


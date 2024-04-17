from config import *
from utils import print_config_options

def handle_lineProfiler():
    """
    When profiling using line_profiler it requires @profile decorator be added to code.
    calling this function before decorating any function with @profile allows one to leave the @profile decorators
    even after the profilig is over.
    """
    try:
        # Python 2
        import __builtin__ as builtins
    except ImportError:
        # Python 3
        import builtins

    try:
        builtins.profile
    except AttributeError:
        # No line profiler, provide a pass-through version
        def profile(func): return func
        builtins.profile = profile



if __name__ == '__main__':
    handle_lineProfiler()

    if args['config_options']:
        print_config_options()
    if args['train']:
        from train import main
    elif args['eval']:
        from eval import main
    main(args)
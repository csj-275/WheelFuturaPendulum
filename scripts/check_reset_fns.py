"""检查 reset_root_state_uniform 的签名和参数。"""
import argparse
from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from isaaclab.envs import mdp
import inspect

funcs = [x for x in dir(mdp) if 'reset' in x.lower() and not x.startswith('_')]
for f in sorted(funcs):
    obj = getattr(mdp, f)
    if callable(obj):
        sig = inspect.signature(obj)
        doc = (obj.__doc__ or '')[:100].replace('\n', ' ')
        print(f"{f}{sig}")
        if doc:
            print(f"  -> {doc}")

simulation_app.close()

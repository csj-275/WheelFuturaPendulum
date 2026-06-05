"""验证 reset_root_state_uniform 是否存在。"""
import argparse
from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# 查看 isaaclab.envs.mdp 中可用的 reset 函数
import isaaclab.envs.mdp as mdp
funcs = [x for x in dir(mdp) if 'reset' in x.lower()]
print("Available reset functions:", sorted(funcs))

# 检查 reset_root_state_uniform 签名
if hasattr(mdp, 'reset_root_state_uniform'):
    import inspect
    print("\nreset_root_state_uniform signature:", inspect.signature(mdp.reset_root_state_uniform))
    print("Doc:", mdp.reset_root_state_uniform.__doc__[:300] if mdp.reset_root_state_uniform.__doc__ else "No doc")

simulation_app.close()

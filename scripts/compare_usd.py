"""对比 WheelFuturaPendulum 和 InertialWheel 的 USD 物理参数。"""
import argparse
from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from pxr import Usd, Gf

files = {
    "WFP": "/home/csj/Desktop/WheelFuturaPendulum/usd/WheelFuturaPendulum.usd",
    "InertialWheel": "/home/csj/Desktop/InertialWheel/usd/InertialWheel.usd",
}

for name, path in files.items():
    print(f"\n{'='*60}")
    print(f"=== {name}: {path}")
    print(f"{'='*60}")
    stage = Usd.Stage.Open(path)

    for prim in stage.Traverse():
        prim_type = prim.GetTypeName()
        display = prim.GetPath()
        has_physics = False
        vals = {}

        for attr in prim.GetAttributes():
            aname = attr.GetName()
            keywords = ['physics', 'mass', 'density', 'stiffness', 'damping',
                        'friction', 'restitution', 'joint', 'drive', 'collision',
                        'rigid', 'body', 'inertia']
            if any(k in aname.lower() for k in keywords):
                has_physics = True
                try:
                    vals[aname] = str(attr.Get())[:60]
                except:
                    vals[aname] = "<err>"

        if has_physics:
            print(f"\n  {display} [{prim_type}]")
            for k, v in vals.items():
                print(f"    {k} = {v}")

simulation_app.close()

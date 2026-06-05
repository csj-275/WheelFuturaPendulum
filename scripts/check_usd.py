"""Check USD physics parameters."""
import argparse
from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import carb
import omni.usd
from pxr import Usd, UsdPhysics, Gf

stage = omni.usd.get_context().get_stage()
if not stage:
    # Open the USD
    from pxr import Sdr, UsdUtils
    stage = Usd.Stage.Open("/home/csj/Desktop/WheelFuturaPendulum/usd/WheelFuturaPendulum.usd")

# Print all prims and their physics properties
print("=== All Prims ===")
for prim in stage.Traverse():
    print(f"\nPrim: {prim.GetPath()}")
    print(f"  Type: {prim.GetTypeName()}")

    # Check physics attributes
    for attr in prim.GetAttributes():
        attr_name = attr.GetName()
        if any(k in attr_name for k in ['physics', 'mass', 'density', 'stiffness', 'damping', 'friction', 'restitution', 'joint']):
            try:
                print(f"  {attr_name} = {attr.Get()}")
            except:
                print(f"  {attr_name} = <unreadable>")

simulation_app.close()

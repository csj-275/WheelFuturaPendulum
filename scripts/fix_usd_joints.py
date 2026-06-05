"""清除 USD 中关节内嵌的 stiffness/damping 驱动参数。"""
import argparse
from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from pxr import Usd, UsdPhysics, Gf

wfp_path = "/home/csj/Desktop/WheelFuturaPendulum/usd/WheelFuturaPendulum.usd"
print(f"Opening: {wfp_path}")
stage = Usd.Stage.Open(wfp_path)

joint_names = ["joint1", "joint2", "wheel_joint"]
for joint_name in joint_names:
    prim_path = f"/World/wheel_urdf/joints/{joint_name}"
    prim = stage.GetPrimAtPath(prim_path)
    if not prim or not prim.IsValid():
        print(f"  Prim {prim_path} not found")
        continue

    print(f"\n  {prim_path} - before:")
    drive = UsdPhysics.DriveAPI.Get(prim, "angular")
    if drive:
        stiffness = drive.GetStiffnessAttr().Get()
        damping = drive.GetDampingAttr().Get()
        max_force = drive.GetMaxForceAttr().Get()
        print(f"    drive:angular:physics:stiffness = {stiffness}")
        print(f"    drive:angular:physics:damping = {damping}")
        print(f"    drive:angular:physics:maxForce = {max_force}")

        # Clear stiffness and damping
        print("  -> Clearing stiffness and damping...")
        drive.GetStiffnessAttr().Set(0.0)
        drive.GetDampingAttr().Set(0.0)
        drive.GetMaxForceAttr().Set(0.0)
    else:
        print("  No angular drive found")

stage.GetRootLayer().Save()
print(f"\nSaved to: {wfp_path}")

# Verify
print("\n=== Verification ===")
stage = Usd.Stage.Open(wfp_path)
for joint_name in joint_names:
    prim_path = f"/World/wheel_urdf/joints/{joint_name}"
    prim = stage.GetPrimAtPath(prim_path)
    drive = UsdPhysics.DriveAPI.Get(prim, "angular") if prim else None
    if drive:
        s = drive.GetStiffnessAttr().Get()
        d = drive.GetDampingAttr().Get()
        print(f"  {joint_name}: stiffness={s}, damping={d}")

simulation_app.close()

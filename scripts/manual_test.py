"""Manual smoke test — verify MuJoCo + robosuite + EGL rendering works."""

import numpy as np
import robosuite as suite

print("=== GraspAnything Manual Test ===")
print(f"robosuite version: {suite.__version__}")

# 1. Create environment with Panda arm
print("\n[1] Creating environment (Panda + Lift)...")

env = suite.make(
    "Lift",
    robots="Panda",
    has_renderer=False,            # no GUI window
    has_offscreen_renderer=True,   # EGL offscreen rendering
    render_camera="frontview",
    use_camera_obs=True,
    camera_depths=True,
    camera_names="agentview",
    control_freq=20,
    horizon=100,
)
print("    ✓ Environment created")

# 2. Reset and get observation
print("\n[2] Resetting environment...")
obs = env.reset()
cam = "agentview"
print(f"    RGB image shape:  {obs[f'{cam}_image'].shape}  (expect H×W×3)")
has_depth = f"{cam}_depth" in obs
print(f"    Depth map:        {'available' if has_depth else 'NOT available — need camera_depths=True'}")
if has_depth:
    print(f"    Depth map shape:  {obs[f'{cam}_depth'].shape}")
    print(f"    Depth min/max:    {obs[f'{cam}_depth'].min():.3f} / {obs[f'{cam}_depth'].max():.3f}")
print(f"    RGB dtype:        {obs[f'{cam}_image'].dtype}")
print("    ✓ Observation OK")

# 3. Run a few random actions
print("\n[3] Running 20 random action steps...")
for i in range(20):
    action = np.random.randn(env.action_dim)
    obs, reward, done, info = env.step(action)
print(f"    Final reward: {reward:.3f}")
print("    ✓ Random actions executed")

# 4. Try a simple reach-and-grasp sequence
print("\n[4] Testing arm movement (move down, close gripper, lift)...")
env.reset()
for step in range(50):
    # Oscillate: open/close gripper, move arm slightly
    action = np.zeros(env.action_dim)
    if step < 20:
        action[2] = -0.2  # move down (dz)
        action[6] = 1.0   # keep gripper open (gripper is dim 6)
    elif step < 35:
        action[6] = -1.0  # close gripper
    else:
        action[2] = 0.3   # lift up (dz)
        action[6] = -1.0  # keep closed
    obs, reward, done, info = env.step(action)

print(f"    Completed {step+1} steps, reward: {reward:.3f}")
print("    ✓ Arm control sequence OK")

# 5. Save sample images
print("\n[5] Saving sample images...")
from PIL import Image

rgb = obs[f"{cam}_image"]
depth = obs.get(f"{cam}_depth", None)

Image.fromarray(rgb).save("test_rgb.png")
if depth is not None:
    depth = np.squeeze(depth)  # (H, W, 1) → (H, W)
    depth_norm = ((depth - depth.min()) / (depth.max() - depth.min()) * 255).astype(np.uint8)
    Image.fromarray(depth_norm).save("test_depth.png")
    print("    ✓ Saved test_rgb.png and test_depth.png")
else:
    Image.fromarray(rgb).save("test_rgb.png")
    print("    ✓ Saved test_rgb.png (no depth available)")

print("\n=== All tests passed! Environment is ready. ===")
env.close()

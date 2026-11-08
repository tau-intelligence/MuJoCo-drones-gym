"""Example: Play/visualize a trained policy.

Usage:
    python -m multi_drone_mujoco.examples.play --model_path results/rl_hover/best_model.zip
    python -m multi_drone_mujoco.examples.play --model_path results/rl_hover/best_model.zip --gui
    python -m multi_drone_mujoco.examples.play --model_path results/rl_hover/best_model.zip --record
"""

import argparse
import numpy as np


def play(model_path: str, env_type: str = "hover", episodes: int = 3,
         gui: bool = False, record: bool = False):
    """Load and visualize a trained policy."""
    try:
        from stable_baselines3 import PPO
    except ImportError:
        print("[ERROR] stable-baselines3 not installed.")
        return

    from multi_drone_mujoco.envs.hover_aviary import HoverAviary
    from multi_drone_mujoco.envs.multi_hover_aviary import MultiHoverAviary

    print(f"Loading model from: {model_path}")
    model = PPO.load(model_path)

    render_mode = "human" if gui else "rgb_array"
    if env_type == "multi":
        env = MultiHoverAviary(num_drones=2, ctrl_freq=48, sim_freq=240,
                               gui=gui, record=record, render_mode=render_mode)
    else:
        env = HoverAviary(ctrl_freq=48, sim_freq=240,
                          gui=gui, record=record, render_mode=render_mode)

    for ep in range(episodes):
        obs, info = env.reset()
        total_reward = 0.0
        steps = 0
        positions = []

        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            positions.append(np.array(env.pos, dtype=float).copy())
            if gui:
                env.render()

            if terminated or truncated:
                break

        positions = np.array(positions)  # (T, num_drones, 3)
        final = positions[-1]
        status = "TERMINATED (crash/out of bounds)" if terminated else "completed"
        print(f"  Episode {ep + 1}: reward={total_reward:.2f}, steps={steps}, {status}")
        for i, p in enumerate(final):
            print(f"    drone {i}: final pos=({p[0]:+.3f}, {p[1]:+.3f}, {p[2]:+.3f})")

    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--env_type", type=str, default="hover")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--gui", action="store_true", help="Open the interactive viewer")
    parser.add_argument("--record", action="store_true", help="Save PNG frames")
    args = parser.parse_args()
    play(args.model_path, args.env_type, args.episodes, gui=args.gui, record=args.record)

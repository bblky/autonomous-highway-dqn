import os
import shutil
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import highway_env
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from moviepy.editor import ImageSequenceClip, VideoFileClip, concatenate_videoclips
from PIL import Image, ImageDraw
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor


# CONFIGURATION

script_location = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_location)
print(f"📍 Working Directory set to: {os.getcwd()}")

LOG_DIR = "./training_logs"
VIDEO_DIR = "./videos"
MODEL_PATH = "highway_dqn_model"
EVOLUTION_FILENAME = "evolution_dqn_dense.mp4"
EVAL_CSV_PATH = os.path.join(LOG_DIR, "eval_results.csv")

TOTAL_TIMESTEPS = 250_000 
EVAL_EPISODES = 30
FPS = 15

ENV_CONFIG: Dict[str, Any] = {
    "observation": {
        "type": "Kinematics",
        "vehicles_count": 15, 
        "features": ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"],
        "features_range": {"x": [-200, 200], "y": [-20, 20], "vx": [-30, 30], "vy": [-20, 20]},
        "absolute": False,
        "normalize": True,
        "order": "sorted",
    },
    "action": {"type": "DiscreteMetaAction", "target_speeds": [0, 7, 15, 22, 30]},
    "lanes_count": 4,

    "vehicles_count": 20, 
    
    "duration": 600,
    "initial_lane_id": None,

    "simulation_frequency": 15,
    "policy_frequency": 5,

    "collision_reward": 0,
    "high_speed_reward": 0, 
    "right_lane_reward": 0, 
    "lane_change_reward": 0,
}


# REWARD WRAPPER

class DenseTrafficReward(gym.Wrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.target_speed = 30.0

        self.w_speed = 1.0
        self.w_risk = 1.2     
        self.w_crash = -100.0
        self.w_lane_center = 0.05 
        self.w_lane_change = -0.05 
        self.ttc_threshold = 2.5
        self.dist_threshold = 25.0 

        self.prev_y: Optional[float] = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        ego = self.env.unwrapped.vehicle
        self.prev_y = float(ego.position[1])
        return obs, info

    def _nearest_vehicle_any_lane(self) -> Tuple[float, float]:
        ego = self.env.unwrapped.vehicle
        ex, ey = float(ego.position[0]), float(ego.position[1])

        min_d = np.inf
        min_lat = np.inf
        for v in self.env.unwrapped.road.vehicles:
            if v is ego: continue
            vx, vy = float(v.position[0]), float(v.position[1])
            d = float(np.hypot(vx - ex, vy - ey))
            if d < min_d:
                min_d = d
                min_lat = abs(vy - ey)
        return min_d, min_lat

    def _front_vehicle_same_lane(self) -> Tuple[float, float]:
        ego = self.env.unwrapped.vehicle
        ex, ey, ev = float(ego.position[0]), float(ego.position[1]), float(ego.speed)

        front_dx = np.inf
        closing = 0.0
        for v in self.env.unwrapped.road.vehicles:
            if v is ego: continue
            dx = float(v.position[0] - ex)
            dy = abs(float(v.position[1] - ey))
            if dy < 2.0 and dx > 0 and dx < front_dx:
                front_dx = dx
                closing = max(0.0, ev - float(v.speed))
        return front_dx, closing

    def step(self, action):
        obs, _, terminated, truncated, info = self.env.step(action)
        ego = self.env.unwrapped.vehicle

        if ego.crashed or (not ego.on_road):
            terminated = True
            return obs, float(self.w_crash), terminated, truncated, info

        # 1. Speed
        speed = float(ego.speed)
        speed_reward = float(np.interp(speed, [0.0, self.target_speed], [0.0, 1.0]))

        # 2. TTC Risk
        front_dx, closing = self._front_vehicle_same_lane()
        ttc_risk = 0.0
        if np.isfinite(front_dx) and closing > 0.1:
            ttc = front_dx / closing
            ttc_risk = float(np.interp(ttc, [0.0, self.ttc_threshold], [1.0, 0.0]))

        # 3. Proximity Risk
        min_dist, min_lat = self._nearest_vehicle_any_lane()
        proximity_risk = 0.0
        if np.isfinite(min_dist):
            proximity_risk = float(np.interp(min_dist, [0.0, self.dist_threshold], [1.0, 0.0]))
        
        # Side-swipe Logic
        side_amp = float(np.interp(min_lat, [0.0, 4.0], [1.0, 0.0]))
        proximity_risk *= side_amp

        risk = max(ttc_risk, proximity_risk)

        # 4. Lane Centering & Fee
        try:
            _, lateral = ego.lane.local_coordinates(ego.position)
            lane_center = self.w_lane_center * float(np.exp(-0.5 * float(lateral) ** 2))
        except:
            lane_center = 0.0

        lane_change_penalty = 0.0
        if self.prev_y is not None:
            if abs(float(ego.position[1]) - float(self.prev_y)) > 1.0:
                lane_change_penalty = self.w_lane_change
        self.prev_y = float(ego.position[1])

        reward = (self.w_speed * speed_reward) - (self.w_risk * risk) + lane_center + lane_change_penalty
        return obs, float(reward), terminated, truncated, info


# HELPER FUNCTIONS

def create_env(render_mode: Optional[str] = None) -> gym.Env:
    env = gym.make("highway-fast-v0", render_mode=render_mode)
    env.unwrapped.configure(ENV_CONFIG)
    env.reset()
    env = DenseTrafficReward(env)
    return env

def add_overlay(frame: np.ndarray, phase_text: str, trial: int, speed: float) -> np.ndarray:
    img = Image.fromarray(frame)
    w, h = img.size
    header_h = 50
    canvas = Image.new("RGB", (w, h + header_h), (20, 20, 20))
    canvas.paste(img, (0, header_h))
    d = ImageDraw.Draw(canvas)
    d.text((15, 15), f"PHASE: {phase_text}", fill=(255, 255, 255))
    d.text((15, 30), f"TRIAL: {trial}/3", fill=(200, 200, 200))
    d.text((w - 160, 20), f"{float(speed):.1f} m/s", fill=(255, 255, 255))
    return np.array(canvas)

def record_phase(model: DQN, phase_name: str, num_trials: int, limit_seconds: int) -> List[str]:
    print(f"\n🎥 Recording Phase: {phase_name} (Limit: {limit_seconds}s)")
    env = create_env(render_mode="rgb_array")
    paths: List[str] = []
    max_steps = limit_seconds * FPS

    for trial in range(1, num_trials + 1):
        obs, _ = env.reset()
        done = False
        truncated = False
        frames: List[np.ndarray] = []
        steps = 0

        while not (done or truncated):
            if phase_name.lower() == "untrained":
                action = env.action_space.sample()
            else:
                action, _ = model.predict(obs, deterministic=True)

            obs, _, done, truncated, _ = env.step(action)
            frame = env.render()
            frames.append(add_overlay(frame, phase_name, trial, env.unwrapped.vehicle.speed))
            steps += 1

            if env.unwrapped.vehicle.crashed:
                crash = Image.fromarray(frames[-1])
                d = ImageDraw.Draw(crash)
                d.text((250, 200), "CRASHED", fill=(255, 0, 0))
                for _ in range(10): frames.append(np.array(crash))
                break

            if steps >= max_steps:
                break

        out = os.path.join(VIDEO_DIR, f"{phase_name.replace(' ', '_')}_{trial}.mp4")
        ImageSequenceClip(frames, fps=FPS).write_videofile(out, logger=None)
        paths.append(out)

    env.close()
    return paths

def evaluate_agent(model: DQN, n_episodes: int, seed: int) -> Dict[str, float]:
    env = create_env(render_mode=None)
    rewards, lengths, speeds = [], [], []
    crashes = 0

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        done = False
        truncated = False
        ep_reward = 0.0
        steps = 0
        ep_speeds: List[float] = []

        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, r, done, truncated, _ = env.step(action)
            ep_reward += float(r)
            steps += 1
            ep_speeds.append(float(env.unwrapped.vehicle.speed))

        rewards.append(ep_reward)
        lengths.append(steps)
        speeds.append(float(np.mean(ep_speeds)) if ep_speeds else 0.0)
        if env.unwrapped.vehicle.crashed:
            crashes += 1

    env.close()
    return {
        "avg_reward": float(np.mean(rewards)),
        "avg_length": float(np.mean(lengths)),
        "crash_rate": float(crashes / n_episodes),
        "avg_speed": float(np.mean(speeds)),
    }

def append_eval_csv(step: int, metrics: Dict[str, float], csv_path: str) -> None:
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    exists = os.path.exists(csv_path)
    with open(csv_path, "a", encoding="utf-8") as f:
        if not exists:
            f.write("step,avg_reward,avg_length,crash_rate,avg_speed\n")
        f.write(f"{step},{metrics['avg_reward']:.4f},{metrics['avg_length']:.2f},"
                f"{metrics['crash_rate']:.4f},{metrics['avg_speed']:.2f}\n")

def plot_training_results(log_dir: str) -> None:
    path = os.path.join(log_dir, "monitor.csv")
    if not os.path.exists(path):
        return
    df = pd.read_csv(path, skiprows=1)
    if "r" not in df.columns: return
    window = max(1, int(len(df) * 0.05))
    df["rolling"] = df["r"].rolling(window=window).mean()

    plt.figure(figsize=(10, 6))
    plt.style.use("bmh")
    plt.plot(df["r"], alpha=0.2, color="gray", label="Episode Reward")
    plt.plot(df["rolling"], linewidth=2, label=f"Avg Reward ({window} eps)")
    plt.title("Agent Training Performance (DQN)")
    plt.xlabel("Episodes")
    plt.ylabel("Cumulative Reward")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(LOG_DIR, "training_curve.png"), dpi=150)
    print("✅ Graph saved.")

def main():
    if os.path.exists(LOG_DIR): shutil.rmtree(LOG_DIR)
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(VIDEO_DIR, exist_ok=True)

    train_env = make_vec_env(lambda: create_env(render_mode=None), n_envs=1, seed=0, vec_env_cls=DummyVecEnv)
    train_env = VecMonitor(train_env, LOG_DIR)

    print("\n🧠 Initializing DQN (Stable + Dense Traffic)...")
    model = DQN(
        "MlpPolicy",
        train_env,
        policy_kwargs=dict(net_arch=[256, 256]),
        learning_rate=1e-4,
        buffer_size=TOTAL_TIMESTEPS + 150_000, 
        learning_starts=15_000,
        batch_size=128,
        gamma=0.99,
        train_freq=4,
        gradient_steps=4,
        target_update_interval=4000,
        exploration_fraction=0.1,    
        exploration_final_eps=0.05,
        verbose=1,
        tensorboard_log=LOG_DIR,
    )

    all_clips: List[str] = []

    print("\n--- 1) Recording Untrained ---")
    all_clips.extend(record_phase(model, "Untrained", 3, 10))

    half = TOTAL_TIMESTEPS // 2
    print(f"\n--- 2) Train Half ({half}) ---")
    model.learn(total_timesteps=half)

    print("\n--- Eval @ Half ---")
    m_half = evaluate_agent(model, EVAL_EPISODES, seed=123)
    print(m_half)
    append_eval_csv(half, m_half, EVAL_CSV_PATH)

    print("\n--- Recording Half-Trained ---")
    all_clips.extend(record_phase(model, f"Half-Trained ({half})", 3, 20))

    print(f"\n--- 3) Train Full (+{half}) ---")
    model.learn(total_timesteps=half, reset_num_timesteps=False)
    model.save(os.path.join(LOG_DIR, MODEL_PATH))

    print("\n--- Eval @ Full ---")
    m_full = evaluate_agent(model, EVAL_EPISODES, seed=999)
    print(m_full)
    append_eval_csv(TOTAL_TIMESTEPS, m_full, EVAL_CSV_PATH)

    print("\n--- Recording Fully Trained ---")
    all_clips.extend(record_phase(model, f"Fully Trained ({TOTAL_TIMESTEPS})", 10, 40))

    print("\n--- Generating evolution video ---")
    clips = [VideoFileClip(p) for p in all_clips]
    concatenate_videoclips(clips, method="compose").write_videofile(
        os.path.join(VIDEO_DIR, EVOLUTION_FILENAME), logger=None
    )

    plot_training_results(LOG_DIR)
    print(f"\n✅ Eval metrics saved to: {EVAL_CSV_PATH}")
    print("✅ PROJECT COMPLETE!")

if __name__ == "__main__":
    main()
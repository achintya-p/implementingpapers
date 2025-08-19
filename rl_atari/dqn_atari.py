import argparse
import random
import time
from collections import deque
from dataclasses import dataclass

import ale_py
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Register ALE environments
gym.register_envs(ale_py)

# -------------------------
# Utils
# -------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def to_tensor_uint8_batch(x: np.ndarray, device):
    # x is uint8 [B, C, H, W] in 0..255
    t = torch.from_numpy(x).to(device=device, dtype=torch.float32) / 255.0
    return t

def obs_to_chw_uint8(obs) -> np.ndarray:
    # obs comes from FrameStack + AtariPreprocessing as LazyFrames HWC (84,84,4) uint8
    arr = np.array(obs, copy=False)  # H, W, C
    if arr.ndim == 3 and arr.shape[-1] == 4:
        arr = np.transpose(arr, (2, 0, 1))  # C, H, W
    elif arr.ndim == 2:
        arr = arr[None, ...]  # 1, H, W
    return arr.astype(np.uint8)

# -------------------------
# Environment factory with canonical wrappers
# -------------------------
def make_env(env_id: str, seed: int):
    # Gymnasium Atari v5 IDs, example: "ALE/Breakout-v5"
    env = gym.make(env_id, render_mode=None, frameskip=1)  # disable frame skipping in base env
    from gymnasium.wrappers import AtariPreprocessing, FrameStackObservation
    # Preprocess as in DQN: grayscale 84, frame-skip 4, max-pool over last 2 frames
    env = AtariPreprocessing(
        env,
        noop_max=30,
        frame_skip=4,
        screen_size=84,
        terminal_on_life_loss=False,   # set True if you want EpisodicLife-like behavior
        grayscale_obs=True,
        grayscale_newaxis=False,
        scale_obs=False,
    )
    env = FrameStackObservation(env, 4)
    env.reset(seed=seed)
    env.action_space.seed(seed)
    return env

# -------------------------
# DQN network (Nature 2015)
# -------------------------
class DQN(nn.Module):
    def __init__(self, num_actions: int):
        super().__init__()
        # Input shape: [B, 4, 84, 84]
        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 512)
        self.fc2 = nn.Linear(512, num_actions)

        # Kaiming init for conv, Xavier for linear
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.constant_(m.bias, 0.0) if m.bias is not None else None
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        # x: [B, 4, 84, 84] in [0,1]
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.reshape(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

# -------------------------
# Replay Buffer
# -------------------------
class ReplayBuffer:
    def __init__(self, capacity: int, obs_shape=(4,84,84)):
        self.capacity = capacity
        self.obs_shape = obs_shape
        self.s = np.empty((capacity, *obs_shape), dtype=np.uint8)
        self.a = np.empty((capacity,), dtype=np.int64)
        self.r = np.empty((capacity,), dtype=np.float32)
        self.d = np.empty((capacity,), dtype=np.bool_)
        self.s2 = np.empty((capacity, *obs_shape), dtype=np.uint8)
        self.idx = 0
        self.full = False

    def add(self, s, a, r, d, s2):
        self.s[self.idx] = s
        self.a[self.idx] = a
        self.r[self.idx] = r
        self.d[self.idx] = d
        self.s2[self.idx] = s2
        self.idx = (self.idx + 1) % self.capacity
        if self.idx == 0:
            self.full = True

    def __len__(self):
        return self.capacity if self.full else self.idx

    def sample(self, batch_size: int):
        assert len(self) >= batch_size
        idxs = np.random.randint(0, len(self), size=batch_size)
        return (
            self.s[idxs],
            self.a[idxs],
            self.r[idxs],
            self.d[idxs],
            self.s2[idxs],
        )

# -------------------------
# Training loop
# -------------------------
@dataclass
class Config:
    env_id: str = "ALE/Breakout-v5"
    seed: int = 1
    total_frames: int = 2_000_000      # full paper uses ~200M frames; try 2M for a demo
    learning_starts: int = 50_000
    buffer_size: int = 1_000_000
    batch_size: int = 32
    gamma: float = 0.99
    train_freq: int = 4                # update every 4 env steps
    target_update_freq: int = 10_000   # in gradient steps
    lr: float = 2.5e-4
    eps_start: float = 1.0
    eps_end: float = 0.1
    eps_decay_frames: int = 1_000_000
    eval_interval_frames: int = 100_000
    eval_episodes: int = 5
    save_path: str = "dqn_atari.pt"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    # set to True to log basic stats
    log: bool = True

def linear_eps(cfg: Config, global_step: int):
    if global_step < cfg.learning_starts:
        return cfg.eps_start
    ratio = min(1.0, (global_step - cfg.learning_starts) / cfg.eps_decay_frames)
    return cfg.eps_start + ratio * (cfg.eps_end - cfg.eps_start)

@torch.no_grad()
def evaluate(env_id, online_net: nn.Module, device, seed, episodes=5):
    env = make_env(env_id, seed=seed + 1000)
    online_net.eval()
    scores = []
    for _ in range(episodes):
        obs, _ = env.reset()
        s = obs_to_chw_uint8(obs)
        done = False
        ep_ret = 0.0
        while not done:
            st = to_tensor_uint8_batch(s[None, ...], device)  # [1,4,84,84]
            q = online_net(st)
            action = int(torch.argmax(q, dim=1).item())
            obs2, r, term, trunc, _ = env.step(action)
            done = term or trunc
            ep_ret += float(r)
            s = obs_to_chw_uint8(obs2)
        scores.append(ep_ret)
    env.close()
    return float(np.mean(scores))

def train(cfg: Config):
    set_seed(cfg.seed)
    env = make_env(cfg.env_id, seed=cfg.seed)
    num_actions = env.action_space.n
    device = torch.device(cfg.device)

    online = DQN(num_actions).to(device)
    target = DQN(num_actions).to(device)
    target.load_state_dict(online.state_dict())

    # RMSProp per DeepMind settings
    # PyTorch RMSprop: alpha is smoothing for squared grad. Use eps=0.01 per paper
    optimizer = torch.optim.RMSprop(online.parameters(), lr=cfg.lr, alpha=0.95, eps=0.01)

    rb = ReplayBuffer(cfg.buffer_size)

    obs, _ = env.reset()
    s = obs_to_chw_uint8(obs)
    global_step = 0
    grad_steps = 0
    episode_return = 0.0
    episode_len = 0
    returns = deque(maxlen=10)

    last_eval = 0
    start_time = time.time()

    while global_step < cfg.total_frames:
        eps = linear_eps(cfg, global_step)
        # ε-greedy
        if random.random() < eps:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                st = to_tensor_uint8_batch(s[None, ...], device)
                q = online(st)
                action = int(torch.argmax(q, dim=1).item())

        obs2, r, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        s2 = obs_to_chw_uint8(obs2)

        # Reward clipping
        rc = np.clip(r, -1.0, 1.0).astype(np.float32)

        rb.add(s, action, rc, done, s2)

        episode_return += float(r)
        episode_len += 1
        global_step += 1
        s = s2

        # On end of episode, reset
        if done:
            returns.append(episode_return)
            if cfg.log and len(returns) == returns.maxlen and global_step % 10_000 == 0:
                fps = int(global_step / max(1e-3, time.time() - start_time))
                print(f"step={global_step} avg_return={np.mean(returns):.1f} eps={eps:.3f} fps={fps}")
            obs, _ = env.reset()
            s = obs_to_chw_uint8(obs)
            episode_return = 0.0
            episode_len = 0

        # Learn
        if len(rb) >= cfg.learning_starts and (global_step % cfg.train_freq == 0):
            ss, aa, rr, dd, ss2 = rb.sample(cfg.batch_size)
            ss = to_tensor_uint8_batch(ss, device)
            ss2 = to_tensor_uint8_batch(ss2, device)
            aa = torch.from_numpy(aa).to(device=device, dtype=torch.long)
            rr = torch.from_numpy(rr).to(device=device, dtype=torch.float32)
            dd = torch.from_numpy(dd.astype(np.float32)).to(device=device)  # 0.0 or 1.0

            # Q(s,a)
            q = online(ss)
            q_sa = q.gather(1, aa.unsqueeze(1)).squeeze(1)

            with torch.no_grad():
                q_next = target(ss2).max(dim=1).values
                target_q = rr + cfg.gamma * (1.0 - dd) * q_next

            loss = F.smooth_l1_loss(q_sa, target_q)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            # optional grad clipping to stabilize early training
            torch.nn.utils.clip_grad_norm_(online.parameters(), 10.0)
            optimizer.step()

            grad_steps += 1
            if grad_steps % cfg.target_update_freq == 0:
                target.load_state_dict(online.state_dict())

        # Eval
        if global_step - last_eval >= cfg.eval_interval_frames:
            last_eval = global_step
            avg_score = evaluate(cfg.env_id, online, device, cfg.seed, episodes=cfg.eval_episodes)
            if cfg.log:
                print(f"[eval] step={global_step} score={avg_score:.1f}")
            torch.save({"model": online.state_dict(),
                        "step": global_step,
                        "avg_score": avg_score}, cfg.save_path)

    env.close()
    print("Training done")
    torch.save({"model": online.state_dict(), "step": global_step}, cfg.save_path)

# -------------------------
# Entry
# -------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="ALE/Breakout-v5")
    parser.add_argument("--frames", type=int, default=2_000_000)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    cfg = Config(env_id=args.env, seed=args.seed, total_frames=args.frames)
    train(cfg)

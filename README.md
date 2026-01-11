# Reinforcement Learning - Autonomous Highway Driving

**Authors:** Barış Balkaya (2105157), Bora Çakmak (2201632), Zelal Helin Akdoğan (2004226)  
**Course:** Introduction to Artificial Intelligence and Expert Systems  
**Frameworks:** Gymnasium, Highway-Env, Stable-Baselines3

## 1. Evolution of the Agent
*Evolution video of the Agent starting from untrained, followed by half-trained and fully trained versions.*


https://github.com/user-attachments/assets/86a8dcbc-8475-435a-8c2f-2e58233242a4



---

## 2. Methodology

### The Goal
The objective was to train an agent in the `highway-fast-v0` environment to maximize speed while avoiding collisions. The agent controls a vehicle in a 4-lane highway with dense traffic.

### The Model Architecture
We utilized **Deep Q-Network (DQN)** from the `stable-baselines3` library. After experimenting with PPO (Proximal Policy Optimization), we found DQN yielded better results for this specific discrete action space.

* **Algorithm:** DQN
* **Policy:** MlpPolicy (Multi-Layer Perceptron)
* **Network Architecture:** `[256, 256]` fully connected layers
* **Optimizer:** Adam (`learning_rate=1e-4`)
* **Buffer Size:** 400,000 transitions (High capacity to prevent forgetting rare crash events)
* **Exploration:** Epsilon-greedy starting at 10% and decaying to 5%.

Below is the final DQN model initialization code used for the training:

```python
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
```

### The Mathematical Reward Function
To solve the "passive driver" problem, we engineered a custom wrapper `DenseTrafficReward`. The total reward $R_t$ at step $t$ is calculated as:

$$R_t = (w_{speed} \cdot R_{speed}) - (w_{risk} \cdot R_{risk}) + R_{center} + P_{lane\_change} + P_{crash}$$

Where:

1.  **Speed Reward ($R_{speed}$):** A linear interpolation mapping the current speed $[0, 30] m/s$ to $[0, 1]$. This encourages maximum velocity.
2.  **Risk Penalty ($R_{risk}$):** To prevent reckless driving, we calculate the maximum of two risk factors:
    * *Time-to-Collision (TTC):* How fast we are approaching the car in front.
    * *Proximity:* Physical distance to the nearest car in *any* lane (handling blind spots).
3.  **Lane Centering ($R_{center}$):** A Gaussian function rewarding the agent for staying in the middle of the lane, improving stability.
4.  **Penalties:**
    * $P_{crash} = -100$: A severe penalty to strictly forbid collisions.
    * $P_{lane\_change} = -0.05$: A minor fee to prevent unnecessary zigzagging.

---

## 3. Training Analysis

The agent was trained for **250,000 timesteps**. Below is the performance analysis based on the cumulative reward per episode.

<img width="1500" height="900" alt="training_curve1" src="https://github.com/user-attachments/assets/f2ffd0c8-6d3e-4cea-a6fb-866f77eb893d" />


### Commentary
The learning curve demonstrates a distinct three-phase progression:

1.  **Initial Instability (Ep 0 - 50):** The graph begins with few high rewarding episodes but experiences a sharp drop around episode 50. This correlates with the replay buffer filling up, moving the agent from pure luck to attempting (and initially failing) to maximize the complex reward function.
2.  **Exploration" (Ep 50 - 800):** For the majority of the training, the reward trend is flat with a very slight upward inclination. During this period, the agent is gathering data, learning the physics of the car, and understanding that crashing is the ultimate failure state. It averages around 100 reward points, representing that it can sometimes get things right and sometimes fail while driving.
3.  **Convergence & Mastery (Ep 800 - 1200):** A dramatic shift occurs after around episode 800. We see a sharp upward trend where the agent "cracks" the code—realizing that high-speed and overtaking is the only way to maximize the function. The reward skyrockets from ~100 to ~1150, ending with a stable, high-performing policy.

---

## 4. Challenges & Failures

Our road to a working model was not a straight line. We iterated through several distinct major issues before arriving at the final result.

### 1. Stuck at Low Speeds

**Problem:** The agent was not speeding up properly. It was sitting at a comfortable speed of 20 m/s, never attempting to overtake other cars.

**Fix:** We altered the speed reward logic. Instead of scaling linearly, we split the reward ratio.

* 0–20 m/s now grants very low reward (0-0.4).
* 20–40 m/s grants high reward (0.4-1.0).

This weighted curve made higher speeds mathematically irresistible to the agent.

### 2. The "Profitable Crash" Issue

**Problem:** The agent was crashing even at the fully trained state.

**Root Cause:** We identified that the crash penalty (-5) was too low. The agent calculated that driving at full speed for half a second generated more profit than the loss from a crash.

**Fix:** We raised the penalty to **-25** (and eventually higher), making survival the absolute priority.

### 3. Complexity Problem & Soft Reset
**Problem:** The agent was tailgating cars dangerously.

**Fix (Attempt 1):** We added a penalty for close proximity to the car in front.

**Unexpected Failure:** The agent exploited this by stopping the car (0 m/s) and waiting for the other cars to leave before driving to ensure no proximity existed.

**Fix (Attempt 2):** We added a penalty for idling and a reward for changing lanes.

**Final Failure:** This created "spaghetti rewards": too many conflicting rules. The agent stopped learning entirely.

**Solution:** We performed a **Soft Reset**. We stripped all confusing penalties and reverted to the basics (Speed + Crash) to rebuild the logic from scratch.

### 4. Lack of Curiosity Issue
**Problem:** The agent found a "comfort zone" and stopped trying new strategies.

**Fix:** We introduced an **Entropy Coefficient (`ent_coef=0.01`)**. This hyperparameter forces the agent to explore random actions occasionally, preventing it from getting stuck in local optimum.

### 5. The "Zig-Zag" Effect
**Problem:** The agent was sticking behind other cars and tailing them at mediocre speeds.

**Fix (Attempt 1):** We added a reward for lane changes to encourage overtaking.

**Unexpected Failure:** The agent started prioritizing lane changes over speed, resulting in constant "zig-zag" driving. The reward value was too high.

**Solution:** We adjusted the lane change reward to be tiny: just enough to suggest it as an option, but not enough to be a primary goal.

### 6. The "Brake Check" Issue
**Problem:** The agent would still occasionally "brake check" (slam on the brakes) for no reason.

**Fix:** We modified the Discrete Action Space to remove low speeds entirely. We restricted the target speeds to `[15, 20, 25, 30]`. By physically removing the option to stop, we forced the agent to keep moving.

### 7. Model Switch (PPO vs. DQN)
**Problem:** Despite all tuning, the **PPO** algorithm remained too conservative. It refused to overtake.

**Pivot:** We switched to **DQN**. The discrete nature of DQN immediately solved the overtaking passivity.

**New Issue:** DQN was reckless: it had a 70% crash rate because it would switch lanes without checking if the target lane was empty (blind spot crashes).

**Final Solution:** We introduced a **Composite Risk Reward**. We updated the reward wrapper to calculate risk based on *both* the car in front AND the cars in adjacent lanes. This forced the agent to wait for a safe gap before merging.

### 8. Action Space Re-Calibration
**Problem:** While removing the "stop" option (in Challenge #6) prevented idling, it introduced a new issue: the agent lacked the ability to decelerate smoothly. It was either "fast" or "brake," leading to jerky movements and rear-end collisions.

**Fix:** We redefined the Discrete Action Space (target speeds) to use more balanced intervals: `[0, 7, 15, 22, 30]`.

**Outcome:** These evenly spaced increments allowed the agent to "downshift" logic (e.g., dropping from 30 to 22 or 15) when approaching traffic, rather than panic-braking or crashing. This significantly improved the agent's ability to anticipate and react to the car in front.

### 9. The Mid-Training Reset Bug
**Problem:** We noticed a bizarre phenomenon in our training graphs. Midway through training (specifically when splitting training into two halves), the performance would plummet to zero, as if the agent had suddenly forgotten everything it had learned.

**Root Cause:** We discovered that calling `model.learn()` for the second half of the training session triggered a default reset of the internal counters. The model was effectively performing a hard reset on its learning progress.

**Fix:** We implemented the **`reset_num_timesteps=False`** argument in the learning function:
`model.learn(total_timesteps=half, reset_num_timesteps=False)`

![Training Curve showing dip and recovery]<img width="1500" height="900" alt="training_curve4" src="https://github.com/user-attachments/assets/a0857e5d-6af5-4040-bd65-35b4a91a0a7e" />


**Outcome:** This ensured the agent retained its "memory" between training sessions, allowing the learning curve to continue upward rather than restarting.

### 10. Persistent High Crash Rates
**Problem:** Despite the switch to DQN, the agent continued to exhibit reckless behavior. Trials revealed that the agent often prioritized maintaining high velocity over safety, accepting a "calculated risk" of collision that resulted in an unacceptably high crash rate.

**Fix:** We escalated the crash penalty to extreme levels, drastically increasing `w_crash` to **-1000**.

**Rationale:** The objective was to create a mathematical boundary where the cost of a collision was insurmountable. By raising the penalty to extreme levels, we intended to force the agent to view crashing as a total failure state that could never be outweighed by any potential speed reward, thereby mandating a safety-first policy.

### 11. The Stability Challenge: Overcoming Catastrophic Forgetting
**The Problem:**
We initially assumed that "longer training = better driving." However, we observed a performance degradation when training extended beyond 280,000 steps.

* **< 250k Steps:** The model was under-trained.
* **> 280k Steps:** The model's crash rate spiked, and reward dropped.

![Training Curve showing forgetting](training_curve_4.png)

**The Diagnosis:**
We identified this not as standard overfitting, but as **Catastrophic Forgetting**.
As the agent improved, it stopped crashing. Consequently, the Replay Buffer (FIFO memory) filled up exclusively with "safe driving" data, pushing out the early "crash experiences." The agent effectively forgot that crashing was painful, leading to reckless behavior late in training.

**The Solution:**
We implemented a multi-layered defense strategy to stabilize the agent's long-term memory and "personality":

1.  **Infinite Memory (The Cure):** We set `buffer_size = TOTAL_TIMESTEPS + 150_000`. By making the buffer larger than the entire training run, we ensured the agent never deletes its early crash experiences. It retains the knowledge of "what not to do" forever.
2.  **Timestep Tuning:** Through analysis of 10 validation videos per checkpoint, we identified **250,000 timesteps** as the convergence "Goldilocks Zone"—where the policy was mature but had not yet begun to drift.
3.  **Hyperparameter Tuning:**
    * **Faster Commitment:** Decreased `exploration_fraction` (**0.3 -> 0.1**) to transition from random discovery to skill perfection sooner.
    * **Deeper Thinking:** Increased `gradient_steps` (**2 -> 4**) to force the model to learn more from each batch of data.
    * **Curing Paralysis:** Lowered the crash penalty (**-1000 -> -100**). The extreme penalty was causing the agent to freeze in fear; the adjusted penalty was sufficient to deter crashing while still encouraging movement.

---

## 5. Results & Evaluation

We looked at the training logs to prove that **250,000 timesteps** was the best place to stop training. We compared the agent's performance halfway through training versus the final result.

### Comparison Table

The table below shows the exact numbers from our testing:

| Metric | Half-Trained (125k Steps) | Fully Trained (250k Steps) | Improvement |
| :--- | :--- | :--- | :--- |
| **Average Reward** | 420.98 | **1054.67** | **+150%** |
| **Average Episode Length** | 551.67 | **1130.57** | **+105%** |
| **Crash Rate** | 83.33% | **36.67%** | **-56%** |
| **Average Speed** | 28.04 m/s | **28.87 m/s** | **+3%** |

### What the Numbers Mean

1.  **Safety First:** The biggest improvement was safety. The **Crash Rate** dropped from roughly 83% to 36%. This proves that the second half of training wasn't about driving faster (speed only went up slightly), but about learning how not to crash.
2.  **Higher Scores:** The **Average Reward** more than doubled (420 $\to$ 1054). Since the speed stayed mostly the same, the higher score comes from the car surviving longer and collecting points for distance traveled.
3.  **Longer Drives:** The **Average Length** of a drive doubled. This confirms the car is no longer crashing immediately. It can now navigate through heavy traffic for much longer without failing.

**Conclusion:** The data confirms that **250,000 timesteps** is the "sweet spot." The agent keeps the high speed (~29 m/s) it learned early on, but it is now smart enough to avoid collisions.

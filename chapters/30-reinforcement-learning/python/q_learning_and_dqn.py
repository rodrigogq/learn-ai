"""Chapter 30 - reinforcement learning: Q-learning on a gridworld, then a deep
Q-network (DQN) on CartPole.

Two agents that learn by DOING, from rewards, with no labeled examples:
  1. tabular Q-LEARNING solves a gridworld (reach the goal, avoid the pit) by
     filling in a table of "how good is each action in each state"; you can
     print the whole table and the learned policy;
  2. a DQN replaces that table with a neural network so the same idea scales to
     CartPole, where the state is continuous and a table is impossible.

Run from the repository root:
    .venv/bin/python chapters/30-reinforcement-learning/python/q_learning_and_dqn.py --quick
    .venv/bin/python chapters/30-reinforcement-learning/python/q_learning_and_dqn.py
"""

import argparse
import random
import sys
from pathlib import Path

import numpy
import torch
from torch import nn

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPOSITORY_ROOT))

from common.device import select_best_available_device  # noqa: E402

# --------------------------------------------------------- gridworld Q-learning

GRID_ROWS, GRID_COLUMNS = 3, 4
GOAL_STATE = (0, 3)      # +1 reward
PIT_STATE = (1, 3)       # -1 reward
ACTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]   # up, down, left, right
ACTION_NAMES = ["^", "v", "<", ">"]


def gridworld_step(state, action_index):
    """Take one action in the gridworld.

    Arguments:
        state: (row, column).
        action_index: 0-3 into ACTIONS.

    Returns (next_state, reward, done). Walking into a wall stays put; reaching
    the goal (+1) or pit (-1) ends the episode.
    """
    delta_row, delta_column = ACTIONS[action_index]
    next_row = min(max(state[0] + delta_row, 0), GRID_ROWS - 1)
    next_column = min(max(state[1] + delta_column, 0), GRID_COLUMNS - 1)
    next_state = (next_row, next_column)
    if next_state == GOAL_STATE:
        return next_state, 1.0, True
    if next_state == PIT_STATE:
        return next_state, -1.0, True
    return next_state, -0.02, False   # small step cost, so the agent hurries


def train_q_learning(episodes, discount=0.95, learning_rate=0.1):
    """Tabular Q-learning: learn Q[state][action] = expected future reward.

    The update is the whole algorithm, and it is beautifully simple:
        Q[s][a] += lr * ( reward + discount * max_a' Q[s'][a']  -  Q[s][a] )
    "nudge my estimate toward (what I just got) + (the best I can do next)."
    Exploration uses epsilon-greedy: act randomly sometimes so every state
    gets tried.
    """
    q_table = {(r, c): [0.0, 0.0, 0.0, 0.0] for r in range(GRID_ROWS) for c in range(GRID_COLUMNS)}
    exploration_rate = 1.0
    for episode in range(episodes):
        state = (2, 0)   # start bottom-left
        for _ in range(50):
            if random.random() < exploration_rate:
                action_index = random.randrange(4)
            else:
                action_index = max(range(4), key=lambda a: q_table[state][a])
            next_state, reward, done = gridworld_step(state, action_index)

            best_next = max(q_table[next_state]) if not done else 0.0
            temporal_difference = reward + discount * best_next - q_table[state][action_index]
            q_table[state][action_index] += learning_rate * temporal_difference

            state = next_state
            if done:
                break
        exploration_rate = max(0.05, exploration_rate * 0.995)   # explore less over time
    return q_table


def print_learned_policy(q_table):
    """Show the greedy action in each cell - the learned 'map to the goal'."""
    print("   learned policy (arrow = best action learned for each cell):")
    for row in range(GRID_ROWS):
        cells = []
        for column in range(GRID_COLUMNS):
            if (row, column) == GOAL_STATE:
                cells.append("GOAL")
            elif (row, column) == PIT_STATE:
                cells.append("PIT ")
            else:
                best_action = max(range(4), key=lambda a: q_table[(row, column)][a])
                cells.append(f" {ACTION_NAMES[best_action]}  ")
        print("      " + "".join(cells))


# --------------------------------------------------------- CartPole DQN

class QNetwork(nn.Module):
    """A small MLP mapping a 4-number CartPole state to 2 action-values."""

    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(4, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, 2),
        )

    def forward(self, states):
        return self.network(states)


def train_dqn(episodes, device):
    """Deep Q-learning on CartPole: the table becomes a network.

    Same core update as tabular Q-learning, but Q is now a network trained by
    regression toward reward + discount * max Q(next). Two standard tricks keep
    it stable: a REPLAY BUFFER (learn from a shuffle of past experiences, not
    just the latest) and a slowly-updated TARGET network for the bootstrap.
    """
    try:
        import gymnasium as gym
    except ImportError:
        print("   (gymnasium not installed - skipping DQN. Install with: uv pip install gymnasium)")
        return

    environment = gym.make("CartPole-v1")
    online_network = QNetwork().to(device)
    target_network = QNetwork().to(device)
    target_network.load_state_dict(online_network.state_dict())
    optimizer = torch.optim.Adam(online_network.parameters(), lr=1e-3)

    replay_buffer = []
    exploration_rate = 1.0
    discount = 0.99
    recent_returns = []

    for episode in range(episodes):
        state, _ = environment.reset(seed=episode)
        episode_return = 0.0
        for _ in range(500):
            if random.random() < exploration_rate:
                action = environment.action_space.sample()
            else:
                with torch.no_grad():
                    action = int(online_network(torch.tensor(state, device=device)).argmax())
            next_state, reward, terminated, truncated, _ = environment.step(action)
            done = terminated or truncated
            replay_buffer.append((state, action, reward, next_state, done))
            if len(replay_buffer) > 10000:
                replay_buffer.pop(0)
            state = next_state
            episode_return += reward

            # Learn from a random minibatch of past experience.
            if len(replay_buffer) >= 64:
                batch = random.sample(replay_buffer, 64)
                states, actions, rewards, next_states, dones = zip(*batch)
                # Stack into single numpy arrays first (far faster than building
                # a tensor from a list of arrays, which PyTorch warns about).
                states = torch.tensor(numpy.array(states), device=device, dtype=torch.float32)
                actions = torch.tensor(actions, device=device).unsqueeze(1)
                rewards = torch.tensor(rewards, device=device, dtype=torch.float32).unsqueeze(1)
                next_states = torch.tensor(numpy.array(next_states), device=device, dtype=torch.float32)
                dones = torch.tensor(dones, device=device, dtype=torch.float32).unsqueeze(1)

                current_q = online_network(states).gather(1, actions)
                with torch.no_grad():
                    best_next_q = target_network(next_states).max(dim=1, keepdim=True).values
                    target_q = rewards + discount * best_next_q * (1 - dones)
                loss = nn.functional.mse_loss(current_q, target_q)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if done:
                break

        exploration_rate = max(0.02, exploration_rate * 0.97)
        if episode % 10 == 0:
            target_network.load_state_dict(online_network.state_dict())
        recent_returns.append(episode_return)
        if episode % 20 == 0 or episode == episodes - 1:
            average = sum(recent_returns[-20:]) / len(recent_returns[-20:])
            print(f"   episode {episode:>3}: last-20 average balance time {average:6.1f} steps "
                  f"(500 = perfect, ~22 = random)")
    environment.close()


def main():
    argument_parser = argparse.ArgumentParser(description=__doc__)
    argument_parser.add_argument("--quick", action="store_true", help="fewer episodes")
    parsed_arguments = argument_parser.parse_args()

    random.seed(42)
    torch.manual_seed(42)

    print("1. Tabular Q-learning on a gridworld (reach GOAL +1, avoid PIT -1)")
    q_table = train_q_learning(500 if parsed_arguments.quick else 2000)
    print_learned_policy(q_table)
    print("   Every arrow points along the shortest safe path to the goal - learned purely")
    print("   from rewards, by trial and error, with no example of the right route.")

    print("\n2. Deep Q-network on CartPole (balance a pole by moving a cart)")
    device = select_best_available_device()
    train_dqn(60 if parsed_arguments.quick else 200, device)
    print("   The table became a network, so the SAME idea handles a continuous state.")
    print("   Watch the balance time climb from ~22 (random) toward 500 (solved).")


if __name__ == "__main__":
    main()

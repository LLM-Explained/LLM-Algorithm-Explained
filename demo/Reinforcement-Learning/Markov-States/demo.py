from __future__ import annotations

import random
from dataclasses import dataclass
from collections import defaultdict


@dataclass(frozen=True)
class EnvState:
    position: int
    key_collected: bool


class DelayedGoalEnv:
    """
    A tiny toy environment to illustrate the idea that a compact Markov state
    can be more useful than full growing history.

    World:
      positions 0..4
      key at position 1
      door at position 3
      goal at position 4 behind the door

    Actions:
      0 = left
      1 = right
      2 = pick_key
      3 = open_door

    The true decision state is compact:
      (position, key_collected)

    But a history-based learner may instead memorize long trajectory strings.
    """

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> EnvState:
        self.position = 0
        self.key_collected = False
        self.door_open = False
        self.t = 0
        return EnvState(self.position, self.key_collected)

    def step(self, action: int):
        self.t += 1

        if action == 0:
            self.position = max(0, self.position - 1)
        elif action == 1:
            self.position = min(4, self.position + 1)
        elif action == 2 and self.position == 1:
            self.key_collected = True
        elif action == 3 and self.position == 3 and self.key_collected:
            self.door_open = True

        reward = 0.0
        done = False

        if self.position == 4 and self.door_open:
            reward = 1.0
            done = True

        if self.t >= 12:
            done = True

        return EnvState(self.position, self.key_collected), reward, done


ACTIONS = [0, 1, 2, 3]
ACTION_NAMES = {0: "left", 1: "right", 2: "pick_key", 3: "open_door"}


def epsilon_greedy(q_table, state_key, eps: float = 0.1) -> int:
    if random.random() < eps or state_key not in q_table:
        return random.choice(ACTIONS)
    values = q_table[state_key]
    best = max(values.values())
    best_actions = [a for a, v in values.items() if v == best]
    return random.choice(best_actions)


def init_action_values():
    return {a: 0.0 for a in ACTIONS}


def q_learning(num_episodes: int, use_markov_state: bool):
    env = DelayedGoalEnv()
    q_table = defaultdict(init_action_values)
    alpha = 0.3
    gamma = 0.95
    eps = 0.1
    success_count = 0

    for _ in range(num_episodes):
        state = env.reset()
        history = []

        while True:
            if use_markov_state:
                state_key = (state.position, state.key_collected)
            else:
                # deliberately bad: use growing history as "state"
                state_key = tuple(history)

            action = epsilon_greedy(q_table, state_key, eps)
            next_state, reward, done = env.step(action)

            history.append((state.position, state.key_collected, action))

            if use_markov_state:
                next_key = (next_state.position, next_state.key_collected)
            else:
                next_key = tuple(history)

            best_next = max(q_table[next_key].values()) if not done else 0.0
            old = q_table[state_key][action]
            q_table[state_key][action] = old + alpha * \
                (reward + gamma * best_next - old)

            state = next_state
            if done:
                if reward > 0:
                    success_count += 1
                break

    return q_table, success_count / num_episodes


def evaluate_policy(q_table, use_markov_state: bool, episodes: int = 200):
    env = DelayedGoalEnv()
    success = 0

    for _ in range(episodes):
        state = env.reset()
        history = []

        while True:
            if use_markov_state:
                state_key = (state.position, state.key_collected)
            else:
                state_key = tuple(history)

            values = q_table[state_key]
            action = max(values, key=values.get)

            next_state, reward, done = env.step(action)
            history.append((state.position, state.key_collected, action))
            state = next_state

            if done:
                if reward > 0:
                    success += 1
                break

    return success / episodes


def main():
    random.seed(0)

    train_episodes = 2000

    q_hist, train_succ_hist = q_learning(
        train_episodes, use_markov_state=False)
    q_markov, train_succ_markov = q_learning(
        train_episodes, use_markov_state=True)

    eval_hist = evaluate_policy(q_hist, use_markov_state=False)
    eval_markov = evaluate_policy(q_markov, use_markov_state=True)

    print("=== Markov state abstraction toy demo ===\n")
    print(f"training episodes: {train_episodes}\n")

    print("History-as-state agent")
    print(f"  training success rate : {train_succ_hist:.3f}")
    print(f"  eval success rate     : {eval_hist:.3f}\n")

    print("Markov-state agent")
    print(f"  training success rate : {train_succ_markov:.3f}")
    print(f"  eval success rate     : {eval_markov:.3f}\n")

    print("Interpretation:")
    print("- The compact Markov state lets RL reuse experience across trajectories.")
    print("- The history-as-state agent fragments experience across many long history strings.")
    print("- This is only a toy illustration, but it matches the paper's high-level intuition.")


if __name__ == "__main__":
    main()

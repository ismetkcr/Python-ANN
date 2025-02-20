import numpy as np

class WindyGrid:
    def __init__(self, rows, cols, start):
        self.rows = rows
        self.cols = cols
        self.i = start[0]
        self.j = start[1]

    def set(self, rewards, actions, probs):
        self.rewards = rewards
        self.actions = actions
        self.probs = probs

def windy_grid():
    g = WindyGrid(3, 4, (2, 0))
    rewards = {(0, 3): 1, (1, 3): -1}
    actions = {
        (0, 0): ('D', 'R'),
        (0, 1): ('L', 'R'),
        (0, 2): ('L', 'D', 'R'),
        (1, 0): ('U', 'D'),
        (1, 2): ('U', 'D', 'R'),
        (2, 0): ('U', 'R'),
        (2, 1): ('L', 'R'),
        (2, 2): ('L', 'R', 'U'),
        (2, 3): ('L', 'U'),
    }
    probs = {
        ((2, 0), 'U'): {(1, 0): 1.0},
        ((2, 0), 'R'): {(2, 1): 1.0},
        ((1, 0), 'U'): {(0, 0): 1.0},
        ((1, 0), 'D'): {(2, 0): 1.0},
        ((0, 0), 'D'): {(1, 0): 1.0},
        ((0, 0), 'R'): {(0, 1): 1.0},
        ((0, 1), 'R'): {(0, 2): 1.0},
        ((0, 2), 'R'): {(0, 3): 1.0},
        ((0, 2), 'D'): {(1, 2): 1.0},
        ((2, 1), 'R'): {(2, 2): 1.0},
        ((2, 2), 'R'): {(2, 3): 1.0},
        ((2, 3), 'U'): {(1, 3): 1.0},
        ((1, 2), 'U'): {(0, 2): 0.5, (1, 3): 0.5},
    }
    g.set(rewards, actions, probs)
    return g

# Initialize
transition_probs = {}
rewards = {}
grid = windy_grid()

# Populate transition_probs and rewards with detailed print statements
for (s, a), v in grid.probs.items():
    print(f"\nProcessing transition for state {s} with action {a}")
    for s2, p in v.items():
        transition_probs[s, a, s2] = p
        rewards[(s, a, s2)] = grid.rewards.get(s2, 0)
        print(f"  Current (s = {s}, a = {a}, s2 = {s2}, p = {p}, reward = {rewards[(s, a, s2)]})")

# Example prints the transition probabilities and rewards
print("\nTransition Probabilities:")
for key, value in transition_probs.items():
    print(f"transition_probs[{key}] = {value}")

print("\nRewards:")
for key, value in rewards.items():
    print(f"rewards[{key}] = {value}")

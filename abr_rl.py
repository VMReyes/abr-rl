import d3rlpy
import numpy as np


if __name__ == "__main__":
    logged_data = np.load("data/mpc_traj.npy")
    states = []
    actions = []
    rewards = []
    terminals = []
    for session in logged_data:
        for step in session:
            states.append(np.array(step[0:18]))
            actions.append(np.array(step[18]))
            rewards.append(np.array(step[19]))
            terminals.append(np.array(step[-1]))
    states = np.array(states)
    actions = np.array(actions).astype(int)
    print(actions)
    rewards = np.array(rewards)
    terminals = np.array(terminals)
    dataset = d3rlpy.dataset.MDPDataset(states, actions, rewards, terminals)
    bcq = d3rlpy.algos.BCQ()
    bcq.fit(dataset, n_steps=1000)
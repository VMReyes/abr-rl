import d3rlpy
import numpy as np

LOGGED_DATA_FILES = ["data/bba_traj.npy",
                     "data/bola_traj.npy",
                     "data/mpc_traj.npy",
                     "data/opt_rate_traj.npy",
                     "data/pess_rate_traj.npy",
                     "data/rate_traj.npy"]

if __name__ == "__main__":
    states = []
    actions = []
    rewards = []
    terminals = []
    for data_filename in LOGGED_DATA_FILES:
        print(f'Loading {data_filename}.')
        logged_data = np.load(data_filename)
        for session in logged_data:
            for step in session:
                states.append(np.array(step[0:19]))
                actions.append(np.array(step[19]))
                rewards.append(np.array(step[20]))
                terminals.append(np.array(step[-1]))
    states = np.array(states)
    actions = np.array(actions).astype(int)
    rewards = np.array(rewards)
    terminals = np.array(terminals)
    dataset = d3rlpy.dataset.MDPDataset(states, actions, rewards, terminals)
    bcq = d3rlpy.algos.DiscreteBCQ()
    bcq.fit(dataset, n_epochs=2)

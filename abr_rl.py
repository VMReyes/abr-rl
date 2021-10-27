import d3rlpy
from d3rlpy import dataset
import numpy as np

LOGGED_DATA_FILES = ["data/bba_traj.npy",
                     "data/bola_traj.npy",
                     "data/mpc_traj.npy",
                     "data/opt_rate_traj.npy",
                     "data/pess_rate_traj.npy",
                     "data/rate_traj.npy"]

def get_mdp_dataset_from_datafiles():
    states = []
    actions = []
    rewards = []
    terminals = []
    best_session_score = 0
    best_session_rewards = []
    for data_filename in LOGGED_DATA_FILES:
        print(f'Loading {data_filename}.')
        logged_data = np.load(data_filename)
        for session in logged_data:
            session_rewards = []
            for step in session:
                states.append(np.array(step[0:19]))
                actions.append(np.array(step[19]))
                rewards.append(np.array(step[20]))
                session_rewards.append(np.array(step[20]))
                terminals.append(np.array(step[-1]))
            if np.sum(session_rewards) > best_session_score:
                best_session_rewards = session_rewards.copy()
                best_session_score = np.sum(session_rewards)
    print(f'Best seen session cumulative reward: {np.sum(best_session_rewards)}.')
    print(f'Best seen session average reward: {np.average(best_session_rewards)}.')

    states = np.array(states)
    actions = np.array(actions).astype(int)
    rewards = np.array(rewards)
    terminals = np.array(terminals)
    dataset = d3rlpy.dataset.MDPDataset(states, actions, rewards, terminals)
    return dataset


if __name__ == "__main__":
    dataset = get_mdp_dataset_from_datafiles()
    bcq = d3rlpy.algos.DiscreteBCQ()
    bcq.fit(dataset, n_epochs=1)

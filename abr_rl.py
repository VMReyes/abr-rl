import d3rlpy
from d3rlpy import dataset
import numpy as np
from env.abr import ABRSimEnv

LOGGED_DATA_FILES = ["data/bba_traj.npy"]
                     #"data/bola_traj.npy",
                     #"data/mpc_traj.npy",
                     #"data/opt_rate_traj.npy",
                     #"data/pess_rate_traj.npy",
                     #"data/rate_traj.npy"]

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

def evaluate_agent(agent):
    # Launch ABR environment
    print('Setting up environment..')
    env = ABRSimEnv()

    # Shorthand for number of actions
    act_len = env.action_space.n

    # Number of traces we intend to run through, more gives us a better evaluation
    num_traces = 10

    for _ in range(num_traces):
        # Done in reset: Randomly choose a trace and starting point in it
        obs, _ = env.reset()
        done = False
        t = 0
        rews = []

        while not done:
            # Choose an action through random policy
            act = agent.predict(obs[np.newaxis, :]).item()

            # Take the action
            next_obs, rew, done, info = env.step(act)
            rews.append(rew)

            # Print some statistics
            print(f'At chunk {t}, the agent took action {act}, and got a reward {rew}')
            print(f'\t\tThe observation was {obs}')

            # Going forward: the next state at point t becomes the current state at point t+1
            obs = next_obs
            t += 1
        print(f'Got cumulative reward: {np.sum(rews)}, average reward: {np.average(rews)}.')


if __name__ == "__main__":
    dataset = get_mdp_dataset_from_datafiles()
    bcq = d3rlpy.algos.DiscreteBCQ()
    bcq.build_with_dataset(dataset)
    #bcq.fit(dataset, n_epochs=1)
    bcq.load_model("/home/vmreyes/abr-rl/d3rlpy_logs/DiscreteBCQ_20211027151005/model_183375.pt")
    evaluate_agent(bcq)

import d3rlpy
from d3rlpy import dataset
import numpy as np
from numpy.lib.function_base import average
from env.abr import ABRSimEnv
import csv

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

def evaluate_agent(agent):
    # Launch ABR environment
    print('Setting up environment..')
    env = ABRSimEnv()

    # Number of traces we intend to run through, more gives us a better evaluation
    num_traces = 10

    cumulative_rewards_history = []
    average_rewards_history = []

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
            #print(f'At chunk {t}, the agent took action {act}, and got a reward {rew}')
            #print(f'\t\tThe observation was {obs}')

            # Going forward: the next state at point t becomes the current state at point t+1
            obs = next_obs
            t += 1
        cumulative_reward = np.sum(rews)
        average_reward = np.average(rews)
        print(f'Got cumulative reward: {cumulative_reward}, average reward: {average_reward}.')
        cumulative_rewards_history.append(cumulative_reward)
        average_rewards_history.append(average_reward)
    return {"algorithm": str(agent),
            "average_cumulative_reward": np.average(cumulative_rewards_history),
            "average_average_reward": np.average(average_rewards_history)}

def sweep_epochs_and_algorithms(epochs_to_try):
    dataset = get_mdp_dataset_from_datafiles()
    bcq = d3rlpy.algos.DiscreteBCQ()
    awr = d3rlpy.algos.DiscreteAWR()
    cql = d3rlpy.algos.DiscreteCQL()
    algorithms = [bcq, awr, cql]
    for algorithm in algorithms:
        for epochs in epochs_to_try:
            algorithm.fit(dataset, n_epochs=epochs)
            eval_result = evaluate_agent(algorithm)
            with open('sweep_results.csv', 'a') as resultsfile:
                results_writer = csv.writer(resultsfile)
                results_writer.writerow([eval_result["algorithm"],
                                         epochs,
                                         eval_result["average_average_reward"],
                                         eval_result["average_cumulative_reward"]])


if __name__ == "__main__":
    sweep_epochs_and_algorithms()

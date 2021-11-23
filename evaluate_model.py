import d3rlpy
from d3rlpy import dataset
import numpy as np
import csv
from env.abr import ABRSimEnv
import matplotlib.pyplot as plt

#MODEL_WEIGHTS = "d3rlpy_logs/DiscreteCQL_20211115220329/model_8950.pt"
MODEL_WEIGHTS = "d3rlpy_logs/DiscreteBCQ_20211115222009/model_8950.pt"

def evaluate_agent(agent):
    # Launch ABR environment
    print('Setting up environment..')
    env = ABRSimEnv()

    # Number of traces we intend to run through, more gives us a better evaluation
    num_traces = 25

    trajectories = [] 

    for _ in range(num_traces):
        # Done in reset: Randomly choose a trace and starting point in it
        obs, _ = env.reset()
        done = False
        t = 0
        rews = []
        metrics = {"action": [], "buffer_length": [], "action_bitrate": []}
        while not done:
            # Choose an action through random policy
            act = agent.predict(obs[np.newaxis, :]).item()
            metrics['buffer_length'].append(obs[10])
            metrics['action'].append(act)
            metrics['action_bitrate'].append(obs[13+act])

            # Take the action
            next_obs, rew, done, info = env.step(act)
            rews.append(rew)

            # Print some statistics
            #print(f'At chunk {t}, the agent took action {act}, and got a reward {rew}')
            #print(f'\t\tThe observation was {obs}')

            # Going forward: the next state at point t becomes the current state at point t+1
            obs = next_obs
            t += 1
        trajectories.append(metrics)
    return trajectories

def plot_trajectory(trajectory):
    buffer_history = trajectory["buffer_length"]
    plt.subplot(311)
    plt.plot(buffer_history)
    plt.title("Buffer Length over Time")

    action_history = trajectory["action"]
    plt.subplot(312)
    plt.plot(action_history)
    plt.title("Action Choice over Time")

    action_bitrate_history = trajectory["action_bitrate"]
    plt.subplot(313)
    plt.plot(action_bitrate_history)
    plt.title("Bitrate Choice over Time")

    plt.tight_layout(pad=1.0)
    plt.savefig("output/latest_trajectory_bcq.png")


if __name__ == "__main__":
    #model = d3rlpy.algos.DiscreteCQL.from_json("d3rlpy_logs/DiscreteCQL_20211115220329/params.json")
    model = d3rlpy.algos.DiscreteBCQ.from_json("d3rlpy_logs/DiscreteBCQ_20211115222009/params.json")
    model.load_model(MODEL_WEIGHTS)
    trajectories = evaluate_agent(model)
    plot_trajectory(trajectories[0])



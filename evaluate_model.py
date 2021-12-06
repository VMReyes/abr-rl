import d3rlpy
from d3rlpy import dataset
from d3rlpy.algos.bcq import BCQ
import numpy as np
import csv

from wget import download
from env.abr import ABRSimEnv
import matplotlib.pyplot as plt

import ray
import ray.rllib.agents.ppo as ppo
from train_baseline import myEnv

from policies import BBAAgent

def evaluate_agent(agent, ppo = False, bba = False):
    # Launch ABR environment
    print('Setting up environment..')
    env = ABRSimEnv()

    # Number of traces we intend to run through, more gives us a better evaluation
    num_traces = 200

    trajectories = []

    for _ in range(num_traces):
        # Done in reset: Randomly choose a trace and starting point in it
        obs, _ = env.reset()
        done = False
        t = 0
        metrics = {"action": [],
                    "buffer_length": [],
                    "action_bitrate": [],
                    "download_time": [],
                    "throughput": [],
                    "reward": []}
        cumulative_reward = 0
        while not done:
            # Choose an action through random policy
            if ppo:
                obs_normalized = obs / np.array([1e6, 1e6, 1e6, 1e6, 1e6,
                                                 1, 1, 1, 1, 1,
                                                 40, 490, 6,
                                                 1e6, 1e6, 1e6, 1e6, 1e6, 1e6])
                act = agent.compute_action(obs_normalized)
            elif bba:
                act = agent.take_action(obs)
            else:
                act = agent.predict(obs[np.newaxis, :]).item()
            metrics["throughput"].append(sum(obs[0:4]) / 5) # avg throughput
            metrics["download_time"].append(sum(obs[4:9]) / 5) # avg download time
            metrics['buffer_length'].append(obs[10])
            metrics['action'].append(act)
            metrics['action_bitrate'].append(obs[13+act])

            # Take the action
            next_obs, rew, done, info = env.step(act)
            #cumulative_reward += rew
            metrics["reward"].append(rew)

            # Print some statistics
            #print(f'At chunk {t}, the agent took action {act}, and got a reward {rew}')
            #print(f'\t\tThe observation was {obs}')

            # Going forward: the next state at point t becomes the current state at point t+1
            obs = next_obs
            t += 1
        trajectories.append(metrics)
    return trajectories

def plot_trajectory(trajectory, agent_name):
    buffer_history = trajectory["buffer_length"]
    plt.subplot(411)
    plt.plot(buffer_history)
    plt.title("Buffer Length over Time")

    #action_history = trajectory["action"]
    #plt.subplot(612)
    #plt.plot(action_history)
    #plt.title("Action Choice over Time")

    action_bitrate_history = trajectory["action_bitrate"]
    plt.subplot(412)
    plt.plot(action_bitrate_history)
    plt.title("Bitrate Choice over Time")


    #download_history = trajectory["download_time"]
    #plt.subplot(614)
    #plt.plot(download_history)
    #plt.title("Download Time over Time")

    throughput_history = trajectory["throughput"]
    plt.subplot(413)
    plt.plot(throughput_history)
    plt.title("Throughput over Time")

    cumulative_reward_history = trajectory["reward"]
    plt.subplot(414)
    plt.plot(cumulative_reward_history)
    plt.title("Cumulative Reward over Time")
    
    plt.tight_layout(pad=1.0)
    plt.savefig("output/latest_trajectory_{}.png".format(agent_name))
    plt.close()


def average_trajectory(trajectories, field):
    avg = [0 for _ in range(490)]
    for t in trajectories:
        buffer_history = t[field]
        for i in range(len(buffer_history)):
            avg[i] += buffer_history[i]
    for i in range(490):
        avg[i] /= len(trajectories)
    
    if field == "throughput" or field == "download_time": # wait for five readings!
        return avg[4:]
    return avg

def plot_trajectory_overlay(trajectories, fields, fname):
    '''
    trajectories: list of [name, trajectories]
    fields: list of field names to be overlayed
    '''

    fig, axs = plt.subplots(len(fields))
    for i in range(len(fields)):
        for t, name in trajectories:
            field = fields[i]
            data = average_trajectory(t, field)
            if field == "throughput" or field == "download_time":
                axs[i].plot([i for i in range(5,491)], data, label=name)
            else:
                axs[i].plot(data,label=name)
            axs[i].set_title("Average {} Over Time".format(field))
            axs[i].legend()
    # fig.legend()
    fig.tight_layout(pad=1.0)
    fig.savefig("output/{}".format(fname))
    
def plot_avg_trajectory_metrics(trajectories, agent_name):
    num_trajectories = len(trajectories)
    avg_buffer_length = [0 for _ in range(490)]
    avg_bitrate_history = [0 for _ in range(490)]
    
    for t in trajectories:
        buffer_history = t["buffer_length"]
        for i in range(len(buffer_history)):
            avg_buffer_length[i] += buffer_history[i]
        action_bitrate_history = t["action_bitrate"]
        for i in range(len(buffer_history)):
            avg_bitrate_history[i] += action_bitrate_history[i]
    
    for i in range(490):
        avg_buffer_length[i] /= num_trajectories
        avg_bitrate_history[i] /= num_trajectories
    

    plt.subplot(211)
    plt.plot(avg_buffer_length)
    plt.title("Average Trajectory Buffer Length over Time")

    plt.subplot(212)
    plt.plot(avg_bitrate_history)
    plt.title("Average Trajectory Bitrate Choice over Time")

    plt.tight_layout(pad=1.0)
    plt.savefig("output/average_trajectory_{}.png".format(agent_name))
    





if __name__ == "__main__":
    cql_model = d3rlpy.algos.DiscreteCQL.from_json("d3rlpy_logs/DiscreteCQL_20211115220329/params.json")
    CQL_MODEL_WEIGHTS = "d3rlpy_logs/DiscreteCQL_20211115220329/model_8950.pt"
    cql_alpha_0_model = d3rlpy.algos.DiscreteCQL.from_json("d3rlpy_logs/DiscreteCQL_20211204135430/params.json")
    CQL_MODEL_ALPHA_0_WEIGHTS = "d3rlpy_logs/DiscreteCQL_20211204135430/model_8950.pt"
    cql_alpha_10_model = d3rlpy.algos.DiscreteCQL.from_json("d3rlpy_logs/DiscreteCQL_20211205153019/params.json")
    CQL_MODEL_ALPHA_10_WEIGHTS = "d3rlpy_logs/DiscreteCQL_20211205153019/model_8950.pt"

    bcq_model = d3rlpy.algos.DiscreteBCQ.from_json("d3rlpy_logs/DiscreteBCQ_20211115222009/params.json")
    BCQ_MODEL_WEIGHTS = "d3rlpy_logs/DiscreteBCQ_20211115222009/model_8950.pt"

    ray.init()
    config = ppo.DEFAULT_CONFIG.copy()
    config["num_gpus"] = 0
    config["num_workers"] = 1
    config["lr"] = 1e-2
    config["lambda"] = 0.96
    config["gamma"] = 0.96
    config["entropy_coeff_schedule"] = [(0, 0.2), (2500*490, 0.0)]
    config["model"]["fcnet_hiddens"] = [64, 32]
    config["model"]["fcnet_activation"] = "relu" 
    config["rollout_fragment_length"] = 490

    trainer = ppo.PPOTrainer(config=config, env=myEnv)
    trainer.load_checkpoint("models/ppo/checkpoint_005201/checkpoint-5201")
    ppo_trajectories = evaluate_agent(trainer, ppo=True)

    cql_model.load_model(CQL_MODEL_WEIGHTS)
    cql_trajectories = evaluate_agent(cql_model)
    bcq_model.load_model(BCQ_MODEL_WEIGHTS)
    #bcq_trajectories = evaluate_agent(bcq_model)
    cql_alpha_0_model.load_model(CQL_MODEL_ALPHA_0_WEIGHTS)
    cql_alpha_0_trajectories = evaluate_agent(cql_alpha_0_model)
    cql_alpha_10_model.load_model(CQL_MODEL_ALPHA_10_WEIGHTS)
    cql_alpha_10_trajectories = evaluate_agent(cql_alpha_10_model)
    bcq_trajectories = evaluate_agent(bcq_model)

    bba_agent = BBAAgent(env = ABRSimEnv())
    bba_trajectories = evaluate_agent(bba_agent, bba=True)

    #plot_trajectory(cql_trajectories[0], "cql")
    #plot_trajectory(bcq_trajectories[0], "bcq")
    #plot_trajectory(ppo_trajectories[0], "ppo")
    # plot_trajectory(bba_trajectories[0], "bba")

    # plot_avg_trajectory_metrics(cql_trajectories, "cql")

    trajectories = [(cql_trajectories, "cql"), (bcq_trajectories, "bcq"), (bba_trajectories, "bba"), (ppo_trajectories, "ppo")]
    plot_trajectory_overlay(trajectories, ["reward","throughput"], "trajectories.png")

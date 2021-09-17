from comet_ml import Experiment, ExistingExperiment
log_comet = True

if log_comet:
    experiment = Experiment(
        api_key="CC3qOVi4obAD5yimHHXIZ24HA",
        project_name="marl-arl",
        workspace="peihongyu",
    )

import warnings
warnings.filterwarnings('ignore')

import numpy as np
np.set_printoptions(threshold=np.inf, linewidth=1000, precision=3, suppress=True)

from tqdm import tqdm
from collections import defaultdict
import time

from bin.algo.ppo_agent import PPO, Memory
from bin.algo.constants import CONSTANTS

from multiagent.environment import MultiAgentEnv
import multiagent.scenarios as scenarios

CONST = CONSTANTS()

# get environment
scenario_name = "simple_spread_room"
# load scenario from script
scenario = scenarios.load(scenario_name + ".py").Scenario()
# create world
world = scenario.make_world()
# create multiagent environment
env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, shared_viewer=False)
env.render()
time.sleep(2)

memory = Memory(CONST.NUM_AGENTS)
rlAgent = PPO(env)

# rlAgent.loadModel("checkpoints/ActorCritic_10000.pt", 1)

NUM_EPISODES = 30000
LEN_EPISODES = 1000
UPDATE_TIMESTEP = 1000
curState = []
newState = []
agent_reward_history = [[]] * CONST.NUM_AGENTS

timestep = 0
loss = None

for episode in tqdm(range(NUM_EPISODES)):
    curVecState = np.array(env.reset())
    curImgState = np.array(env.render())
    curState = [curVecState, curImgState]

    agent_episode_reward = [0] * CONST.NUM_AGENTS

    for step in range(LEN_EPISODES):
        timestep += 1

        # Get agent actions
        # for i in range(CONST.NUM_AGENTS):
        #     action = rlAgent.policy.act(curState[i], memory,i)
        #     aActions.append(action)
        aActions = rlAgent.policy_old.act(curState, memory, CONST.NUM_AGENTS)
        # print(aActions)

        # Perform actions
        newVecState, reward, done, info = env.step(aActions)
        newVecState = np.array(newVecState)
        newImgState = np.array(env.render())
        newState = [newVecState, newImgState]
        if step == LEN_EPISODES - 1:
            done = True

        memory.rewards.append(reward)
        memory.is_terminals.append(done)

        if timestep % UPDATE_TIMESTEP == 0:
            loss = rlAgent.update(memory)
            memory.clear_memory()
            timestep = 0

        # record history
        for i in range(CONST.NUM_AGENTS):
            agent_episode_reward[i] += reward[i]

        # set current state for next step
        curState = newState

        if done:
            break

    # post episode

    # Record history
    for i in range(CONST.NUM_AGENTS):
        agent_reward_history[i].append(agent_episode_reward[i])
        if len(agent_reward_history[i]) >= 100:
            agent_reward_history[i] = agent_reward_history[i][-100:]

    if log_comet:
        for i in range(CONST.NUM_AGENTS):
            experiment.log_metric("Reward_agent" + str(i), agent_episode_reward[i], episode)
        experiment.log_metric("EpLen", step + 1, episode)
        if len(agent_reward_history[0]) >= 100:
            for i in range(CONST.NUM_AGENTS):
                experiment.log_metric("Average reward_agent" + str(i), sum(agent_reward_history[i])/100, episode)

    # You may want to plot periodically instead of after every episode
    # Otherwise, things will be slow
    # rlAgent.summaryWriter_addMetrics(episode, loss, reward_history, agent_history_dict, LEN_EPISODES)
    if episode % 50 == 0:
        rlAgent.saveModel("checkpoints")

    if episode % 1000 == 0:
        rlAgent.saveModel("checkpoints", True, episode)

rlAgent.saveModel("checkpoints")
env.out.release()
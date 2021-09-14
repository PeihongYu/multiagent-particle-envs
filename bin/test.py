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
reward_history = []
agent_history_dict = defaultdict(list)

timestep = 0
loss = None

for episode in tqdm(range(NUM_EPISODES)):
    curVecState = env.reset()
    curImgState = env.render()
    curState = [curVecState, curImgState]

    episodeReward = 0
    epidoseLoss = 0
    episodeNewVisited = 0
    episodePenalty = 0
    agent_episode_reward = [0] * CONST.NUM_AGENTS

    for step in range(LEN_EPISODES):
        timestep += 1

        # Get agent actions
        # for i in range(CONST.NUM_AGENTS):
        #     action = rlAgent.policy.act(curState[i], memory,i)
        #     aActions.append(action)
        aActions = rlAgent.policy_old.act(curState, memory, CONST.NUM_AGENTS)

        # Perform actions
        newVecState, reward, done, info = env.step(aActions)
        newImgState = env.render()
        newState = [newVecState, newImgState]
        if step == LEN_EPISODES - 1:
            done = True

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
    reward_history.append(episodeReward)

    for i in range(CONST.NUM_AGENTS):
        agent_history_dict[i].append((agent_episode_reward[i]))

    # You may want to plot periodically instead of after every episode
    # Otherwise, things will be slow
    rlAgent.summaryWriter_addMetrics(episode, loss, reward_history, agent_history_dict, LEN_EPISODES)
    if episode % 50 == 0:
        rlAgent.saveModel("checkpoints")

    if episode % 1000 == 0:
        rlAgent.saveModel("checkpoints", True, episode)

rlAgent.saveModel("checkpoints")
env.out.release()
import numpy as np
from multiagent.core import World, Agent, Landmark, Wall
from multiagent.scenario import BaseScenario
from multiagent.scenarios.room_arguments import get_room_args

room_args = get_room_args()

class Scenario(BaseScenario):
    def make_world(self):
        world = World()
        # add agents
        world.agents = [Agent() for i in range(1)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = True
            agent.silent = True
        # add landmarks
        world.landmarks = [Landmark() for i in range(1)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = True
            landmark.movable = False

        # world.walls = [Wall() for i in range(2)]
        # # add walls
        # for i, wall in enumerate(world.walls):
        #     wall.name = 'wall %d' % i
        #     wall.collide = True
        #     wall.movable = False

        world.walls = [Wall() for i in range(room_args.wall_num)]
        # add walls
        for i, wall in enumerate(world.walls):
            wall.name = 'wall %d' % i
            wall.collide = True
            wall.movable = False

        # make initial conditions
        self.reset_world(world)
        return world

    def reset_world(self, world):
        # random properties for agents
        for i, agent in enumerate(world.agents):
            agent.color = np.array([0.25,0.25,0.25])
        # random properties for landmarks
        for i, landmark in enumerate(world.landmarks):
            landmark.color = np.array([0.75,0.75,0.75])
        # random properties for walls
        for i, wall in enumerate(world.walls):
            wall.color = np.array([0, 0.7, 0.0])
        world.landmarks[0].color = np.array([0.75,0.25,0.25])
        # set random initial states
        for agent in world.agents:
            agent.state.p_pos = np.array([-0.9, -0.9])
                # np.random.uniform(-1,+1, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)

        landmark_locations = np.zeros((2, 4))
        landmark_locations[:, 0] = np.array([-0.9, -0.9])
        landmark_locations[:, 1] = np.array([-0.9, 0.9])
        landmark_locations[:, 2] = np.array([0.9, -0.2])
        landmark_locations[:, 3] = np.array([0.9, 0.22])
        # for i, entity in enumerate(world.entities):
        #     if 'agent' in entity.name: continue
        #     entity.state.p_pos = landmark_locations[:,i]
        #         # np.random.uniform(-1,+1, world.dim_p)
        #     entity.state.p_vel = np.zeros(world.dim_p)

        for i, landmark in enumerate(world.landmarks):
            landmark.state.p_pos = landmark_locations[:, 1]
                # np.random.uniform(-1,+1, world.dim_p)
            landmark.state.p_vel = np.zeros(world.dim_p)

        for i, wall in enumerate(world.walls):
            wall.state.p_pos = np.array(room_args.wall_centers[i]) + 0.8
            wall.state.p_vel = np.zeros(world.dim_p)
            wall.x_len = room_args.wall_shapes[i][0]
            wall.y_len = room_args.wall_shapes[i][1]

    def reward(self, agent, world):
        dist2 = np.sum(np.square(agent.state.p_pos - world.landmarks[0].state.p_pos))
        return -dist2

    def observation(self, agent, world):
        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for entity in world.landmarks:
            entity_pos.append(entity.state.p_pos - agent.state.p_pos)
        return np.concatenate([agent.state.p_vel] + entity_pos)

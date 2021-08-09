"""
Code for an agent in the microworld
"""
import numpy as np
import torch
from agents import HillClimbingAgent
from Microworld_experiment import Microworld


class MicroworldMacroAgent:
    """
    An agent in the simulated microworld.
    This class acts as a wrapper for other agent classes.
    """
    def __init__(self, A=None, B=None, init_endogenous=None, subgoal_dimensions=None,
                 init_exogenous=None, T=None, final_goal=None, lr=None, cost=None,
                 von_mises_parameter=None, exponential_parameter=None, step_with_model=False,
                 exo_cost=None, continuous_attention=False, verbose=True):
        """
        Parameters --
        A, B, init_endogenous, von_mises_parameter, exponential_parameter: parameters for the simulated microworld
        subgoal_dimensions: the number of dimensions the subgoal should have
        init_exogenous: the initial states of the exogenous variables
        T: total amount of time for the agent to run
        final_goal: the final goal of the agent
        lr: not sure about this yet
        cost: not sure
        step_with_model: whether to use the noise model on each step
        verbose: whether to print things out
        """
        # location and scale of the final goal
        self.final_goal_loc = final_goal[0]
        self.final_goal_scale = torch.diag(1 / final_goal[1])
        # whether to take noisy steps
        self.step_with_model = step_with_model
        # variables to store statistics about the model runs
        self.final_goal_dist_all = []
        self.closeness_all = []
        self.final_goal_reached_all = []
        self.all_endogenous = []
        self.final_reached_t = 0
        # parameters for running the model
        self.T = T
        self.verbose = verbose

        # create a microworld environment
        self.env = Microworld(A=A, B=B, init=init_endogenous, von_mises_parameter=von_mises_parameter,
                              exponential_parameter=exponential_parameter)

        # compute the distance between the initial state and the goal location
        self.initial_dist = self.distance(self.env.endogenous_state, self.final_goal_loc, self.final_goal_scale)

        # set up a hill-climbing agent
        self.agent = HillClimbingAgent(A=A, B=B, goal_loc=self.final_goal_loc, goal_scale=self.final_goal_scale,
                                       initial_dist=self.initial_dist, subgoal_dimensions=subgoal_dimensions,
                                       init_exogenous=init_exogenous, step_size=lr,
                                       continuous_attention=continuous_attention, att_cost=cost,
                                       exo_cost=exo_cost)

    def distance(self, endogenous_state, loc, scale):
        """
        Compute euclidean distance between endogenous_state and loc, scaled by the matrix scale
        """
        return torch.sqrt(torch.matmul(torch.matmul((endogenous_state - loc).float(), scale.float()).unsqueeze(0),
                                       (endogenous_state - loc).float().t()))

    def step(self, stop_t=None):
        """
        Do one step with this agent, selecting an action and updating the state of the microworld. Always takes at least
        one step, otherwise keeps taking steps until stop_t is reached.
        """
        # previous state
        s_prev = self.env.endogenous_state

        # initialize the done indicator to false
        done = False
        # keep looping until either reaching an intermediate goal or running out of time
        while not done:
            # check if out of time
            if self.agent.t >= self.T:
                done = True
            else:
                done = False

            # increment time by 1
            self.agent.t += 1

            # print out information if in verbose mode
            if self.verbose:
                print('                                                                           ')
                print('                                                                           ')
                print('                                                                           ')
                print('Timestep ---------------------------------------------------', self.agent.t)

            # get the agent's action
            agent_exo = self.agent.select_sparsemax_action(self.env)

            # take a step in the microworld
            if self.step_with_model:
                self.env.step_with_model(agent_exo)
            else:
                self.env.step(agent_exo)

            # get distance between final goal and current state
            final_goal_dist = self.distance(self.env.endogenous_state, self.final_goal_loc, self.final_goal_scale)

            # we have achieved the final goal if we are within 30 of it (From an older version with binary achievement)
            if final_goal_dist <= 30:
                final_goal_indicator = 1
            else:
                final_goal_indicator = 0

            if self.verbose:
                print('Current goal distance -----------------------------------------------',
                      final_goal_dist.tolist()[0])

            # set the time that the agent reached the goal if it reached the goal on this step
            if final_goal_indicator and self.final_reached_t == 0:
                self.final_reached_t = self.agent.t

            # print state, action, and goal if in verbose mode
            if self.verbose:
                print('Choosen exogenous action -------------------------------------------',
                      self.agent.exogenous.tolist())

            if self.verbose:
                print('Endogenous state ---------------------------------------', self.env.endogenous_state.tolist())

            if self.verbose:
                print('Goal -----------------------------------', self.agent.goal_loc.tolist())

            # append current step's information to traces
            self.all_endogenous.append(self.env.endogenous_state.tolist())
            self.final_goal_dist_all.append(float(final_goal_dist))
            self.final_goal_reached_all.append(final_goal_indicator)
            self.closeness_all.append(self.initial_dist - final_goal_dist.detach())

            # if it is currently the stopping time specified in the parameter, break
            if stop_t <= self.agent.t:
                break

        s_next = self.env.endogenous_state

        return s_prev, s_next, done

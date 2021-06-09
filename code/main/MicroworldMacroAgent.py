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
    def __init__(self, A=None, B=None, init_endogenous=None, subgoal_dimensions=None, nr_subgoals=0,
                 init_exogenous=None, T=None, final_goal=None, lr=None, clamp=None, cost=None, agent_class=None,
                 von_mises_parameter=None, exponential_parameter=None, step_with_model=False, decision=None,
                 exo_cost=None, use_exo_cost=False, continuous_attention=False, use_input_cost=False,
                 input_cost=False, decision_type=None, verbose=True):
        """
        Parameters --
        A, B, init_endogenous, von_mises_parameter, exponential_parameter: parameters for the simulated microworld
        subgoal_dimensions: the number of dimensions the subgoal should have
        nr_subgoals: the number of subgoals to compute
        init_exogenous: the initial states of the exogenous variables
        T: total amount of time for the agent to run
        final_goal: the final goal of the agent
        lr: not sure about this yet
        clamp: not sure
        cost: not sure
        agent_class: the type of agent
        step_with_model: whether to use the noise model on each step
        decision: not sure yet, fits into sparsemax model
        verbose: whether to print things out
        """
        self.final_goal = final_goal
        self.final_goal_loc = final_goal[0]
        self.final_goal_scale = torch.diag(1 / final_goal[1])
        self.step_with_model = step_with_model
        self.subgoal_dimensions = subgoal_dimensions
        self.nr_subgoals = nr_subgoals
        self.cost = cost
        self.final_reached_t = 0
        self.last_subgoal_t = 0
        self.last_subgoal_cost = 0
        self.subgoals_reached_t = []
        self.final_goal_dist_all = []
        self.closeness_all = []
        self.agent_class = agent_class
        self.costs_all = []
        self.final_goal_reached_all = []
        self.all_endogenous = []
        self.T = T
        self.decision = decision
        self.verbose = verbose

        # create a microworld environment
        self.env = Microworld(A=A, B=B, init=init_endogenous, von_mises_parameter=von_mises_parameter,
                              exponential_parameter=exponential_parameter, agent="hillclimbing")

        # compute the distance between the initial state and the goal location
        self.initial_dist = self.distance(self.env.endogenous_state, self.final_goal_loc, self.final_goal_scale)

        # set up a hill-climbing agent
        self.agent = HillClimbingAgent(A=A, B=B, final_goal_loc=self.final_goal_loc,
                                       final_goal_scale=self.final_goal_scale,
                                       initial_dist=self.initial_dist, subgoal_dimensions=subgoal_dimensions,
                                       init_exogenous=init_exogenous, init_endogenous=init_endogenous, lr=lr,
                                       clamp=clamp, continuous_attention=continuous_attention, cost=self.cost,
                                       exo_cost=exo_cost, use_exo_cost=use_exo_cost, use_input_cost=use_input_cost,
                                       input_cost=input_cost, decision_type=decision_type)
        self.final_subgoal_t = self.agent.t
        # if there are no subgoals, configure the agent to pursue the final goal directly.
        if nr_subgoals <= 1:
            self.agent.goal_loc = self.final_goal_loc
            self.agent.goal_scale = self.final_goal_scale

    def distance(self, endogenous_state, loc, scale):
        """
        Compute euclidean distance between endogenous_state and loc, scaled by the matrix scale
        """
        return torch.sqrt(torch.matmul(torch.matmul((endogenous_state - loc).float(), scale.float()).unsqueeze(0),
                                       (endogenous_state - loc).float().t()))

    def step(self, stop_t=None):
        """
        Do one step with this agent, selecting an action and updating the state of the microworld.
        """
        # previous state
        s_prev = self.env.endogenous_state

        # set indicator to say we haven't reached an intermediate goal
        intermediate_goal_indicator = False

        # if no more subgoals
        if self.nr_subgoals == -1 and self.agent.t == 0:
            working_on_final = True
        else:
            working_on_final = False

        done = False

        # keep looping until either reaching an intermediate goal or running out of time
        while not intermediate_goal_indicator and not done:
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

            # set subgoal dimensions to all if working on final dimension
            if working_on_final:
                self.agent.subgoal_dimensions = [0, 1, 2, 3, 4]

            self.agent.select_sparsemax_action(self.env, decision=self.decision)

            # take a microworld step
            if self.step_with_model:
                self.env.step_with_model(self.agent.exogenous)
            else:
                self.env.step(self.agent.exogenous)

            # get endogenous state based on subgoal dimensions (attention)
            lower_dim_endogenous_state = torch.tensor([entry.item() for i, entry in enumerate(self.env.endogenous_state)
                                                       if i in self.agent.subgoal_dimensions])

            # get distance between final goal and current state
            final_goal_dist = self.distance(self.env.endogenous_state, self.final_goal_loc, self.final_goal_scale)

            # git distance between current subgoal (with attention) and current goal
            current_goal_dist = self.distance(lower_dim_endogenous_state, self.agent.goal_loc, self.agent.goal_scale)

            # we have achieved the final goal if we are within 30 of it
            if final_goal_dist <= 30:
                final_goal_indicator = 1
            else:
                final_goal_indicator = 0

            # we have achieved the current goal if we are within 30 of it
            if current_goal_dist <= 30:  # / (1 / float((torch.sum(torch.tensor(goal[1])))))
                intermediate_goal_indicator = 1
            else:
                intermediate_goal_indicator = 0

            # even if we've achieved the current goal, pretend we haven't if we're working toward the final goal
            if intermediate_goal_indicator and working_on_final:
                intermediate_goal_indicator = 0

            # print the number of edges ignored and the current goal distance if in verbose mode
            # if self.verbose:
                # print('Nr of edges ignored -----------------------------------------------',
                #       self.agent.nr_of_edges_ignored[-1].item())
            if self.verbose:
                print('Current goal distance -----------------------------------------------',
                      current_goal_dist.tolist()[0])

            # set the time that the agent reached the goal if it reached the goal on this step
            if final_goal_indicator and self.final_reached_t == 0 and working_on_final:
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

            if intermediate_goal_indicator and self.nr_subgoals > 0:

                self.subgoals_reached_t.append(self.agent.t)

            # if we have completed the last subgoal, switch to working on final goal
            if intermediate_goal_indicator and self.nr_subgoals > 0 and not working_on_final:
                self.subgoals_reached_t.append(self.agent.t)
                intermediate_goal_indicator = False
                working_on_final = True
                self.nr_subgoals -= 1
                self.agent.goal_loc = self.final_goal_loc
                self.agent.goal_scale = self.final_goal_scale
                self.final_subgoal_t = self.agent.t
                self.last_subgoal_cost = np.sum(self.costs_all)
                self.last_subgoal_t = self.agent.t
                self.subgoal_dimensions = [0, 1, 2, 3, 4]
                self.agent.subgoal_dimensions = [0, 1, 2, 3, 4]
                if self.verbose:
                    print('----------------------------- SWITCH TO FINAL ----------------------------------',)

            # if it is currently the stopping time specified in the parameter, break
            if stop_t is None or self.agent.t is None:
                print("None issue!")
            if stop_t <= self.agent.t:
                break

        s_next = self.env.endogenous_state

        return s_prev, s_next, done

    def goal_sample(self):
        """
        Come up with a new goal by sampling a point in space using uniform and normal distributions
        """
        m = torch.distributions.Uniform(torch.tensor([-250.0]), torch.tensor([450.0]))
        n = torch.distributions.Normal(torch.tensor([3.]), torch.tensor([20.]))
        loc = m.sample()
        scale = torch.abs(1 / n.sample())

        goal = torch.tensor([[loc], [scale]])

        return goal

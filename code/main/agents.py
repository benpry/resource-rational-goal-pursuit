"""
Defines the classes for human and hill climbing agents
The human agent is a class that takes input from a human trying to achieve the goal.
"""
import numpy as np
import torch
from Microworld_experiment import Microworld
from graded_attention import analytic_attention


class HillClimbingAgent:
    """ An agent class that can work towards a goal in a simulated microworld. This class contains implementations
    of the hill-climbing and sparse hill-climbing agents with both continuous and discrete attention.
    """

    def __init__(self, A=None, B=None, goal_loc=None, goal_scale=None, initial_dist=None,
                 subgoal_dimensions=None, att_cost=None, init_exogenous=None, step_size=None,
                 continuous_attention=False, exo_cost=None):
        """
        Initialize this hill-climbing agent
        goal_loc: the location of the currently-pursued goal
        goal_scale: the scale of the currently pursued goal (for this paper, the scale is all 1s)
        initial_dist: the initial distance from the goal
        subgoal_dimensions: the dimensions to pay attention to on the current subgoal (for compatibility with future work)
        att_cost: cost of attention for testing attention vectors
        init_exogenous: initial exogenous input (generally all 0s)
        step_size: the multiple of the optimal step size to use
        continuous_attention: whether attention should be continuous or discrete
        exo_cost: the 'c' value determining the weight of exogenous costs
        """
        if A is not None:
            self.A = A
        if B is not None:
            self.B = B
        # configure the initial distance, goal location, and goal scale
        self.initial_dist = initial_dist
        self.goal_loc = goal_loc
        self.goal_scale = goal_scale
        self.subgoal_dimensions = subgoal_dimensions
        # make sure init_exogenous is a tensor
        if type(init_exogenous) == torch.Tensor:
            self.exogenous = init_exogenous
        else:
            self.exogenous = torch.tensor(init_exogenous, dtype=torch.float64)

        # the parameters of the model
        self.continuous_attention = continuous_attention
        self.att_cost = att_cost
        self.step_size = step_size

        # the parameter of the environment
        self.exo_cost = exo_cost

        # variables to keep track of model behaviour
        self.nr_of_edges_ignored = []
        self.nr_of_edges_attended = []
        self.all_exogenous = []
        self.exogenous_cost = 0.
        self.t = 0

    def distance(self, endogenous_state, loc, scale):
        """
        euclidean distance scaled by the scale argument
        """
        return torch.sqrt(torch.matmul(torch.matmul((endogenous_state - loc), scale).unsqueeze(0),
                                       (endogenous_state - loc).t()))

    def create_edges(self, subgoal_dimensions):
        """
        Generate a list of all the links between variables in the simulated microworld
        subgoal_dimensions: list[int] - a list of indices of the variables to pay attention to
        """

        # set up the transition matrices for the microworld
        A = torch.tensor([[1., 0., 0., 0., 0.], [0., 1., 0., 0., -0.5], [0., 0., 1., 0., -0.5],
                          [0.1, -0.1, 0.1, 1., 0.], [0., 0., 0., 0.0, 1.]])
        B = torch.tensor([[0., 0., 2., 0.], [5., 0., 0., 0.], [3., 0., 5., 0.], [0., 0., 0., 2.],
                          [0., 10., 0., 0.]])

        # cut out rows and columns of matrices that aren't part of the subgoal being pursued
        A = A.index_select(0, torch.tensor(subgoal_dimensions))
        A = A.index_select(1, torch.tensor(subgoal_dimensions))
        B = B.index_select(0, torch.tensor(subgoal_dimensions))

        # come up with links between endogenous variables
        possible_links_a, possible_links_b = [], []
        for i in range(len(A)):
            for j in range(len(A)):
                if A[i, j] != 0. and i != j:
                    possible_links_a.append(['var_source', i, j])

        # come us with links between exogenous and endogenous variables
        for i in range(len(B)):
            for j in range(4):
                if B[i, j] != 0.:
                    possible_links_b.append(['var_exo', i, j])

        all_comb = possible_links_a + possible_links_b

        return all_comb

    def create_attention_mv(self, endogenous, attention_vector):
        """
        Create a representation of a microworld with the specified attention vector
        endogenous: the initial endogenous state
        attention_vector: a list specifying which edges to ignore
        """
        # create copies of A and B so as not to modify them permanently
        A = self.A.clone().detach()
        B = self.B.clone().detach()

        # set the links specified by the attention vector to 0
        for i in attention_vector:
            if i[0] == 'var_source':
                A[i[1], i[2]] = 0.

            elif i[0] == 'var_exo':
                B[i[1], i[2]] = 0.

        # take only the dimensions of the subgoal that the agent pays attention to
        A = A.index_select(0, torch.tensor(self.subgoal_dimensions))
        A = A.index_select(1, torch.tensor(self.subgoal_dimensions))
        B = B.index_select(0, torch.tensor(self.subgoal_dimensions))

        # select the parts of the endogenous state that correspond to the subgoal dimensions
        if len(endogenous) > 1:
            init_endogenous = [entry.item() for i, entry in enumerate(endogenous.squeeze(0))
                               if i in self.subgoal_dimensions]
        else:
            init_endogenous = endogenous

        # set up a microworld with A and B
        env = Microworld(A=A, B=B, init=init_endogenous)

        return env

    def select_sparsemax_action(self, microworld):
        """
        Select an action according to the sparsemax model
        microworld: the microworld in which the agent exists
        """

        if self.continuous_attention:
            # select the dimensions relevant to the subgoal (if a subgoal is being pursued)
            A = microworld.A.index_select(0, torch.tensor(self.subgoal_dimensions))
            A = A.index_select(1, torch.tensor(self.subgoal_dimensions))
            B = microworld.B.index_select(0, torch.tensor(self.subgoal_dimensions))
            state = microworld.endogenous_state.index_select(0, torch.tensor(self.subgoal_dimensions))

            # create a microworld for the current subgoal
            subgoal_microworld = Microworld(A, B, init=state)

            # choose an exogenous action using analytic attention
            best_exogenous, performance, total_ignorance = analytic_attention(subgoal_microworld, self.goal_scale,
                                                                              self.goal_loc, self.att_cost,
                                                                              self.step_size,
                                                                              exo_cost=self.exo_cost)
            # set the exogenous variable
            self.exogenous = best_exogenous
            # save performance statistics
            self.nr_of_edges_ignored.append(total_ignorance)
            self.nr_of_edges_attended.append(len(torch.nonzero(A)) + len(torch.nonzero(B)) - A.shape[0] - total_ignorance)
            self.exogenous_cost = self.exo_cost * self.exogenous.dot(self.exogenous)
            self.all_exogenous.append(self.exogenous)
        else:  # discrete attention version
            # generate a list of all the edges in the attention-reduced microworld
            list_of_edges = self.create_edges(self.subgoal_dimensions)
            num_edges = len(list_of_edges)
            # initialize arrays for finding the best attention vector
            best_all_sizes, best_all_sizes_performance, best_exogenous_all_sizes = [], [], []

            # get the best attention vector by enumerating over all possible binary vectors
            best_attention_vector = []
            for i, attention_vector_size in enumerate(range(num_edges + 1)):
                if attention_vector_size == 0:
                    # try the empty attention vector, which attends to everything
                    best_exogenous, performance = self.test_attention_vector([], microworld)
                else:
                    # get the best attention vector with size k, along with its performance and edges
                    best_attention_vector, performance, list_of_edges, best_exogenous = \
                        self.find_best_attention_vector_of_size_k(best_attention_vector, list_of_edges, microworld)
                # deepcopy
                best_all_sizes.append(best_attention_vector[:])
                best_all_sizes_performance.append(performance)
                best_exogenous_all_sizes.append(best_exogenous)

            # choose the best-performing attention vector among the bests of each size
            choice = np.nanargmax(best_all_sizes_performance)
            self.nr_of_edges_ignored.append(len(best_all_sizes[choice]))
            self.nr_of_edges_attended.append(num_edges - len(best_all_sizes[choice]))
            self.exogenous = best_exogenous_all_sizes[choice]
            self.exogenous_cost = self.exo_cost * self.exogenous.dot(self.exogenous)
            self.all_exogenous.append(self.exogenous)

        return self.exogenous

    def find_best_attention_vector_of_size_k(self, best_attention_vector, list_of_edges, microworld):
        """
        Find the best attention vector of size k. I.e. if you can only pay attention to k relationships, which
        should they be? This only works if k > 0
        best_attention_vector: the best attention vector one size smaller
        list_of_edges: a list of all the edges that could be attended to
        microworld: the microworld in which the agent operates
        """
        performance_all_new_edges, all_exogenous = [], []

        # try adding each possible remaining edge to the attention vector and measure the performance
        for new_edge in list_of_edges:
            # test the resulting attention vector
            exogenous, performance = self.test_attention_vector(best_attention_vector, microworld, new_edge=new_edge)

            # add the performance and action to the list of existing states and actions
            performance_all_new_edges.append(performance)
            all_exogenous.append(exogenous)

        # get the best edge, performance, and attention vector
        best_edge = np.nanargmax(performance_all_new_edges)
        best_edge_performance = np.nanmax(performance_all_new_edges)
        best_attention_vector.append(list_of_edges[best_edge])
        list_of_edges.pop(best_edge)
        best_exogenous = all_exogenous[best_edge]
        return best_attention_vector, best_edge_performance, list_of_edges, best_exogenous

    def test_attention_vector(self, best_attention_vector, microworld, new_edge=None):
        """
        Test the attention vector specified by "best_attention_vector" with the edge new_edge added
        best_attention_vector: the best attention vector prior to the new edge being added
        microworld: the microworld the agent operates in
        new_edge: the new edge we are considering adding to the representation of the microworld
        """

        # add the new edge to the attention vector
        if new_edge:
            best_attention_vector.append(new_edge)
        test_attention_vector = best_attention_vector[:]  # deep copy
        # save the current endogenous state (to reset it later)
        endogenous = microworld.endogenous_state

        # ignore the parts of the microworld specified by the test attention vector
        microworld_attention = self.create_attention_mv(endogenous, test_attention_vector)

        # take a step in the reduced microworld with a zero input (to account for endogenous connections)
        microworld_attention.step(torch.zeros(4, dtype=torch.float64))

        # get distance from current goal
        default_dist = self.distance(microworld_attention.endogenous_state, self.goal_loc, self.goal_scale)

        # compute gradient of distance between current location and current goal state
        gradient = -(torch.matmul(torch.div(
            torch.matmul(self.goal_scale, (microworld_attention.endogenous_state - self.goal_loc)), default_dist),
            microworld_attention.B))

        if gradient.norm() == 0:
            opt_step_size = 0
        else:
            opt_step_size = - self.goal_scale.inverse()\
                .matmul(microworld_attention.endogenous_state - self.goal_loc)\
                .dot(microworld_attention.B.mv(gradient)) / (microworld_attention.B.mv(gradient)
                                                             .matmul(self.goal_scale.inverse())
                                                             .dot(microworld_attention.B.mv(gradient))
                                                             + self.exo_cost * (gradient.dot(gradient)))

        exogenous = self.step_size * opt_step_size * gradient

        # take one step in the real microworld with the exogenous action and compute the distance to the goal
        microworld.step(exogenous)
        goal_dist = self.distance(microworld.endogenous_state, self.goal_loc, self.goal_scale)

        # reset the microworld's endogenous state (undoing the last step)
        microworld.endogenous_state = endogenous

        # compute number of edges the agent attends to
        non_zero_connections = microworld_attention.A.flatten() != 0.
        self_connections = microworld_attention.A.flatten() != 1.
        cost_edges = float(torch.sum(non_zero_connections * self_connections) +
                           torch.sum(microworld_attention.B.flatten() != 0.))

        # compute performance (change in cost from the initial state, minus an attention cost)
        performance = self.initial_dist - torch.sqrt(goal_dist**2 + self.exo_cost * exogenous.dot(exogenous))\
            - self.att_cost * cost_edges

        # remove the new edge that was added to the attention vector
        if new_edge:
            best_attention_vector.pop(-1)

        return exogenous, performance

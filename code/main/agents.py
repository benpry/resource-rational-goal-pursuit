"""
Defines the classes for human and hill climbing agents
The human agent is a class that takes input from a human trying to achieve the goal.
"""
import numpy as np
import torch
from Microworld_experiment import Microworld
from graded_attention import analytic_attention


class HillClimbingAgent:
    """ An agent class that can work towards a goal in a simulated microworld. Uses the sparse-max hill-climbing
     model
    """

    def __init__(self, A=None, B=None, final_goal_loc=None, final_goal_scale=None, goal_loc=None, goal_scale=None, initial_dist=None,
                 subgoal_dimensions=None, cost=None, init_exogenous=None, init_endogenous=None, lr=None, clamp=None,
                 continuous_attention=False, exo_cost=None, use_exo_cost=False, verbose=True, use_input_cost=False,
                 input_cost=None, decision_type=None):
        """
        Initialize this hill-climbing agent
        final_goal_loc: the location of the final goal the agent pursues
        final_goal_scale: the permitted distance from the goal that the agent can have
        goal_loc: the location of the currently-pursued goal
        initial_dist: the initial distance from the goal
        subgoal_dimensions: the dimensions to pay attention to on the current subgoal
        cost: cost of distance from goal for testing attention vectors
        init_exogenous: initial exogenous state
        lr: step size (like learninig rate) when
        clamp: Gradient clipping +-clamp (budget constraint)
        """
        if A is not None:
            self.A = A
        if B is not None:
            self.B = B
        self.initial_dist = initial_dist
        self.final_goal_loc = final_goal_loc
        self.final_goal_scale = final_goal_scale
        self.goal_loc = goal_loc
        self.goal_scale = goal_scale
        self.subgoal_dimensions = subgoal_dimensions
        if type(init_exogenous) == torch.Tensor:
            self.exogenous = init_exogenous
        else:
            self.exogenous = torch.tensor(init_exogenous, dtype=torch.float64)
        self.lr = lr  # the step size of the model
        self.exogenous_cost = 0.
        self.clamp = clamp
        self.all_exogenous = []
        self.all_edges, self.endogenous_self_edges = self.create_edges(self.subgoal_dimensions)
        self.verbose = True
        self.t = 0
        self.cost = cost
        self.nr_of_edges_ignored = []
        self.nr_of_edges_attended = []
        self.continuous_attention = continuous_attention
        self.exo_cost = exo_cost
        self.use_exo_cost = use_exo_cost
        self.use_input_cost = use_input_cost
        self.input_cost = input_cost
        if type(init_endogenous) != torch.Tensor:
            init_endogenous = torch.tensor(init_endogenous)
        self.starting_cost = init_endogenous.dot(init_endogenous)
        self.decision_type = decision_type

    def distance(self, endogenous_state, loc, scale):
        """
        euclidean distance scaled by the scale argument
        """
        return torch.sqrt(torch.matmul(torch.matmul((endogenous_state - loc), scale).unsqueeze(0),
                                       (endogenous_state - loc).t()))

    def create_edges(self, subgoal_dimensions):
        """
        Generate a list of all the links, with
        subgoal_dimensions: list[int] - a list of indices of the variables to pay attention to
        """

        # set up the microworld with matrices
        A = torch.tensor([[1., 0., 0., 0., 0.], [0., 1., 0., 0., -0.5], [0., 0., 1., 0., -0.5],
                          [0.1, -0.1, 0.1, 1., 0.], [0., 0., 0., 0.0, 1.]])
        B = torch.tensor([[0.0, 0.0, 2., 0.], [5., 0., 0., 0.], [3., 0., 5., 0.], [0., 0., 0., 2.],
                          [0., 10., 0., 0.]])

        # cut out rows and columns of matrices that the agent isn't focusing on
        A = A.index_select(0, torch.tensor(subgoal_dimensions))
        A = A.index_select(1, torch.tensor(subgoal_dimensions))
        B = B.index_select(0, torch.tensor(subgoal_dimensions))

        # come up with links between reduced endogenous states
        possible_links_a, possible_links_b = [], []
        for i in range(len(A)):
            for j in range(len(A)):
                if A[i, j] != 0. and i != j:
                    possible_links_a.append(['var_source', i, j])

        # come us with links between exogenous ande endogenous variables
        for i in range(len(B)):
            for j in range(4):
                if B[i, j] != 0.:
                    possible_links_b.append(['var_exo', i, j])

        all_comb = possible_links_a + possible_links_b

        return all_comb, possible_links_a

    def create_attention_mv(self, endogenous, attention_vector):
        """
        Create a representation of a microworld with the specified attention vector
        endogenous: the initial endogenous state
        attention_vector:
        """
        A = self.A.clone().detach()
        B = self.B.clone().detach()

        for i in attention_vector:
            if i[0] == 'var_source':
                A[i[1], i[2]] = 0.

            elif i[0] == 'var_exo':
                B[i[1], i[2]] = 0.

        # take only the dimensions of the subgoal that the agent pays attention to
        A = A.index_select(0, torch.tensor(self.subgoal_dimensions))
        A = A.index_select(1, torch.tensor(self.subgoal_dimensions))
        B = B.index_select(0, torch.tensor(self.subgoal_dimensions))
        if len(endogenous) > 1:
            init_endogenous = [entry.item() for i, entry in enumerate(endogenous.squeeze(0))
                               if i in self.subgoal_dimensions]
        else:
            init_endogenous = endogenous

        # set up a microworld with A and B
        env = Microworld(A=A, B=B, init=init_endogenous, agent="hillclimbing")

        return env

    def select_sparsemax_action(self, microworld, decision='softmax', temp=1):
        """
        Select an action according to the sparsemax model
        microworld: the microworld in which the agent exists
        decision: the decision function
        temp: the temprature?
        """

        if self.continuous_attention:
            A = microworld.A.index_select(0, torch.tensor(self.subgoal_dimensions))
            A = A.index_select(1, torch.tensor(self.subgoal_dimensions))
            B = microworld.B.index_select(0, torch.tensor(self.subgoal_dimensions))
            state = microworld.endogenous_state.index_select(0, torch.tensor(self.subgoal_dimensions))

            subgoal_microworld = Microworld(A, B, init=state, agent='hillclimbing')

            best_exogenous, performance, total_ignorance = analytic_attention(subgoal_microworld, self.goal_scale,
                                                                              self.goal_loc, self.t, self.cost, self.lr,
                                                                              self.clamp,
                                                                              use_exo_cost=self.use_exo_cost,
                                                                              exo_cost=self.exo_cost,
                                                                              decision_type=self.decision_type)
            self.nr_of_edges_ignored.append(total_ignorance)
            self.nr_of_edges_attended.append(len(torch.nonzero(A)) + len(torch.nonzero(B)) - A.shape[0] - total_ignorance)
            self.exogenous = best_exogenous
            self.exogenous_cost = torch.sum(self.exogenous)
            self.all_exogenous.append(self.exogenous)
        else:
            # generate a list of all the edges in the attention-reduced microworld
            list_of_edges, possible_links_a = self.create_edges(self.subgoal_dimensions)
            num_edges = len(list_of_edges)
            # initialize arrays for finding the best attention vector
            best_all_sizes, best_all_sizes_performance, best_exogenous_all_sizes = [], [], []

            # get the best attention vector by enumerating over all possible binary vectors
            # Plan going forward is to replace this with an analytic solution where attention is graded
            best_attention_vector = []
            for i, attention_vector_size in enumerate(range(1, len(list_of_edges) + 1)):
                if i == 0:
                    best_attention_vector = []
                # get the best attention vector with size i, along with its performance, edges, etc.
                best_attention_vector, performance, list_of_edges, best_exogenous = \
                    self.find_best_attention_vector_of_size_k(i, best_attention_vector, list_of_edges, microworld)
                # deepcopy
                best_all_sizes.append(best_attention_vector[:])
                best_all_sizes_performance.append(performance)
                best_exogenous_all_sizes.append(best_exogenous)

            # this is for backwards-compatibility with an old version of the model that selects attention vectors with
            # a softmax.
            if decision == 'softmax':
                normalized_perf = np.array(best_all_sizes_performance) - np.max(best_all_sizes_performance)
                probs = np.exp(normalized_perf / temp)
                probs /= probs.sum()
                choice = np.random.choice(len(probs), 1, p=probs)[0]
                self.exogenous = best_exogenous_all_sizes[choice]
                self.exogenous_cost = torch.sum(self.exogenous)
                self.all_exogenous.append(self.exogenous)
            else:
                choice = np.nanargmax(best_all_sizes_performance)
                self.nr_of_edges_ignored.append(len(best_all_sizes[choice]))
                self.nr_of_edges_attended.append(num_edges - len(best_all_sizes[choice]))
                self.exogenous = best_exogenous_all_sizes[choice]
                self.exogenous_cost = torch.sum(self.exogenous)
                self.all_exogenous.append(self.exogenous)

    def find_best_attention_vector_of_size_k(self, iteration, best_attention_vector, list_of_edges, microworld):
        """
        Find the best attention vector of size k. I.e. if you can only pay attention to k relationships, which
        should they be?
        iteration: the "k", i.e. the number of edges to attend to
        best_attention_vector
        list_of_edges: a list of all the edges that could be attended to
        microworld: the microworld in which the agent operates
        """
        performance_all_new_edges, all_exogenous = [], []

        # keep trying to add an edge and see how it affects the performance
        for new_edge in list_of_edges:
            # Want first edge to be an edge between edogenous and endogenous
            if iteration == 0 and new_edge[0] == 'var_source':
                continue
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
        test_attention_vector = best_attention_vector[:]
        endogenous = microworld.endogenous_state

        # focus on only the parts of the microworld the agent is paying attention to
        microworld_attention = self.create_attention_mv(endogenous, test_attention_vector)
        # take a step in the reduced microworld
        microworld_attention.step(self.exogenous)

        # get distance from current goal
        out = self.distance(microworld_attention.endogenous_state, self.goal_loc, self.goal_scale)

        # compute gradient of distance between current location and current goal state
        gradient = -(torch.matmul(torch.div(
            torch.matmul(self.goal_scale, (microworld_attention.endogenous_state - self.goal_loc)), out),
            self.t * microworld_attention.B))

        if self.use_exo_cost:
            if gradient.norm() == 0:
                opt_step_size = 0
            else:
                opt_step_size = - self.goal_scale.inverse()\
                    .matmul(microworld_attention.endogenous_state - self.goal_loc)\
                    .dot(microworld_attention.B.mv(gradient)) / (microworld_attention.B.mv(gradient)
                                                                 .matmul(self.goal_scale.inverse())
                                                                 .dot(microworld_attention.B.mv(gradient))
                                                                 + self.exo_cost * (gradient.dot(gradient)))

            exogenous = opt_step_size * self.lr * gradient

        else:
            exogenous = self.lr * gradient
            exogenous_c = torch.sum(torch.abs(exogenous))
            if exogenous_c > self.clamp:
                exogenous = self.clamp / exogenous_c * exogenous

        # take one step with the optimal exogenous variables
        microworld.step(exogenous)

        # check distance from final goal
        final_goal_dist = self.distance(microworld.endogenous_state, self.final_goal_loc, self.final_goal_scale)

        # reset the microworld's endogenous state (undoing the last step)
        microworld.endogenous_state = endogenous

        # compute number of non-zero connections and connections between endogenous variables
        non_zero_connections = microworld_attention.A.flatten() != 0.
        self_connections = microworld_attention.A.flatten() != 1.

        # get the number of edges that the agent attends to
        cost_edges = float(torch.sum(non_zero_connections * self_connections) +
                           torch.sum(microworld_attention.B.flatten() != 0.))

        # compute performance (delta in distance to final goal)
        if self.use_exo_cost:
            performance = self.initial_dist - torch.sqrt(final_goal_dist**2 + self.exo_cost * exogenous.dot(exogenous))\
                          - self.cost * cost_edges
        else:
            performance = (self.initial_dist - final_goal_dist) - self.cost * cost_edges

        # remove the new edge that was added to the attention vector
        best_attention_vector.pop(-1)

        return exogenous, performance

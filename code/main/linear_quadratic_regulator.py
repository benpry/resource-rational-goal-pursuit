"""
Code for the optimal (LQR) model and the sparse attention (sparse LQR) model.
"""
import torch
import numpy as np
from Microworld_experiment import Microworld


class OptimalAgent:
    """
    An agent that uses a linear quadratic regulator to pursue a goal

    A: endogenous transition matrix
    B: exogenous input matrix
    endogenous: the current endogenous state of the system
    Q: the cost of the endogenous state
    Qf: the cost of the final endogenous state
    R: the cost of the exogenous inputs
    """
    A: torch.Tensor
    B: torch.Tensor
    endogenous: torch.Tensor
    Q: torch.Tensor
    Qf: torch.Tensor
    R: torch.Tensor
    T: int
    opt_u: list

    def __init__(self, A, B, Q, Qf, R, T, init_endogenous):
        """
        Initialize this agent.

        A: endogenous transition matrix
        B: exogenous input matrix
        endogenous: the current endogenous state of the system
        Q: the cost of the endogenous state
        Qf: the cost of the final endogenous state
        R: the cost of the exogenous inputs
        """
        self.A = A
        self.B = B
        self.Q = Q
        self.Qf = Qf
        self.R = R
        self.T = T
        self.endogenous = init_endogenous

    def get_actions(self):
        """
        Compute the optimal sequence of actions by backward induction via dynamic programming

        This uses the DP algorithm from slide 23 of these slides:
        https://stanford.edu/class/ee363/lectures/dlqr.pdf
        """
        # n is the number of timesteps the agent has
        n = self.T
        # initialize array of P matrices
        P = [0 for _ in range(n + 1)]

        # iteratively compute the value matrices
        P[n] = self.Qf
        for t in range(n, 0, -1):
            P[t - 1] = self.Q + torch.mm(torch.mm(self.A.t(), P[t]), self.A) - torch.mm(torch.mm(
                torch.mm(torch.mm(self.A.t(), P[t]), self.B),
                (self.R + torch.mm(self.B.t(), torch.mm(P[t], self.B))).inverse()),
                torch.mm(self.B.t(), torch.mm(P[t], self.A)))

        # iteratively compute the optimal action matrices
        K = []
        for t in range(n):
            Kt = -torch.mm((self.R + torch.mm(torch.mm(self.B.t(), P[t+1]), self.B)).inverse(),
                           torch.mm(torch.mm(self.B.t(), P[t+1]), self.A))
            K.append(Kt)

        # compute the list of optimal actions
        u = []
        curr_x = self.endogenous
        for t in range(n):
            u.append(torch.mv(K[t], curr_x))
            curr_x = torch.mv(self.A, curr_x) + torch.mv(self.B, u[-1])
            # prints the timestep, action, and resulting state after each action

        # returns the optimal sequence of actions
        return u


class SparseLQRAgent:
    """
    An agent that uses a linear quadratic regulator to pursue a goal

    A: endogenous transition matrix
    B: exogenous input matrix
    endogenous: the current endogenous state of the system
    Q: the cost of the endogenous state
    Qf: the cost of the final endogenous state
    R: the cost of the exogenous inputs
    attention_cost: the attention cost
    """
    A: torch.Tensor
    B: torch.Tensor
    endogenous: torch.Tensor
    Q: torch.Tensor
    Qf: torch.Tensor
    R: torch.Tensor
    T: int
    opt_u: list

    def __init__(self, A, B, Q, Qf, R, T, init_endogenous, attention_cost):
        """
        Initialize this agent.

        A: endogenous transition matrix
        B: exogenous input matrix
        endogenous: the current endogenous state of the system
        Q: the cost of the endogenous state
        Qf: the cost of the final endogenous state
        R: the cost of the exogenous inputs
        """
        self.A = A
        self.B = B
        self.Q = Q
        self.Qf = Qf
        self.R = R
        self.T = T
        self.endogenous = init_endogenous
        self.attention_cost = attention_cost

    def create_edges(self):
        """
        Create a set of edges representing the system's dynamics
        :return:
        :rtype:
        """
        # come up with links between reduced endogenous states
        possible_links_a, possible_links_b = [], []
        for i in range(len(self.A)):
            for j in range(len(self.A)):
                if self.A[i, j] != 0. and i != j:
                    possible_links_a.append(['var_source', i, j])

        # come us with links between exogenous ande endogenous variables
        for i in range(len(self.B)):
            for j in range(4):
                if self.B[i, j] != 0.:
                    possible_links_b.append(['var_exo', i, j])

        all_comb = possible_links_a + possible_links_b

        return all_comb, possible_links_a

    def create_attention_mv(self, attention_vector):
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

        # set up a microworld with A and B
        env = Microworld(A=A, B=B, init=self.endogenous, agent="hillclimbing")

        return env

    def compute_optimal_sequence(self, microworld):
        """
        Compute the optimal sequence of actions by backward induction via dynamic programming

        This uses the DP algorithm from slide 23 of these slides:
        https://stanford.edu/class/ee363/lectures/dlqr.pdf
        """
        # n is the number of timesteps the agent has
        n = self.T
        # initialize array of P matrices
        P = [0 for _ in range(n + 1)]

        A = microworld.A
        B = microworld.B
        endogenous = microworld.endogenous_state

        # iteratively compute the value matrices
        P[n] = self.Qf
        for t in range(n, 0, -1):
            P[t - 1] = self.Q + torch.mm(torch.mm(A.t(), P[t]), A) - torch.mm(torch.mm(
                torch.mm(torch.mm(A.t(), P[t]), B),
                (self.R + torch.mm(B.t(), torch.mm(P[t], B))).inverse()),
                torch.mm(B.t(), torch.mm(P[t], A)))

        # iteratively compute the optimal action matrices
        K = []
        for t in range(n):
            Kt = -torch.mm((self.R + torch.mm(torch.mm(B.t(), P[t+1]), B)).inverse(),
                           torch.mm(torch.mm(B.t(), P[t+1]), A))
            K.append(Kt)

        # compute the list of optimal actions
        u = []
        curr_x = endogenous
        for t in range(n):
            u.append(torch.mv(K[t], curr_x))
            curr_x = torch.mv(A, curr_x) + torch.mv(B, u[-1])
            # prints the timestep, action, and resulting state after each action

        # returns the optimal sequence of actions
        return u

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
        microworld_attention = self.create_attention_mv(test_attention_vector)

        action_sequence = self.compute_optimal_sequence(microworld_attention)

        for i in range(len(action_sequence)):
            microworld.step(action_sequence[i])

        final_state = microworld.endogenous_state

        microworld.endogenous_state = endogenous

        full_cost = final_state.matmul(self.Qf).dot(final_state)\
            + sum([a.matmul(self.R).dot(a) for a in action_sequence])

        # compute number of non-zero connections and connections between endogenous variables
        non_zero_connections = microworld_attention.A.flatten() != 0.
        self_connections = microworld_attention.A.flatten() != 1.

        # get the number of edges that the agent attends to
        cost_edges = float(torch.sum(non_zero_connections * self_connections) +
                           torch.sum(microworld_attention.B.flatten() != 0.))

        cost = np.sqrt(full_cost) + self.attention_cost * cost_edges

        best_attention_vector.pop(-1)

        return action_sequence, cost

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
            # test the resulting attention vector
            action_sequence, cost = self.test_attention_vector(best_attention_vector, microworld, new_edge=new_edge)

            # add the performance and action to the list of existing states and actions
            performance_all_new_edges.append(cost)
            all_exogenous.append(action_sequence)

        # get the best edge, performance, and attention vector
        best_edge = np.nanargmin(performance_all_new_edges)
        best_edge_performance = np.nanmin(performance_all_new_edges)
        best_attention_vector.append(list_of_edges[best_edge])
        list_of_edges.pop(best_edge)
        best_exogenous = all_exogenous[best_edge]
        return best_attention_vector, best_edge_performance, list_of_edges, best_exogenous

    def choose_opt_attention_vector(self):
        """
        get the best attention vector in the microworld

        :return:
        :rtype:
        """
        list_of_edges, possible_links_a = self.create_edges()

        best_all_sizes, best_all_sizes_performance, best_exogenous_all_sizes = [], [], []
        best_attention_vector = []

        microworld = Microworld(A=self.A, B=self.B, init=self.endogenous)

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

        choice = np.nanargmin(best_all_sizes_performance)
        exogenous = best_exogenous_all_sizes[choice]

        return choice, exogenous

    def get_actions(self):
        """
        compute the optimial action sequence

        :return:
        :rtype:
        """
        best_attention_vector, best_action_sequence = self.choose_opt_attention_vector()

        return best_action_sequence


if __name__ == "__main__":

    # test case computing the optimal action sequence with the sparse LQR agent.
    A = torch.tensor([[1., 0., 0., 0., 0.], [0., 1., 0., 0., -0.5], [0., 0., 1., 0., -0.5],
                      [0.1, -0.1, 0.1, 1., 0.], [0., 0., 0., 0., 1.]])
    B = torch.tensor([[0., 0., 2., 0.], [5., 0., 0., 0.], [3., 0., 5., 0.], [0., 0., 0., 2.], [0., 10., 0., 0.]])
    init_endogenous = torch.tensor([200., 20., 100., 50., -10.])

    Q = torch.tensor([[0., 0., 0., 0., 0.], [0., 0., 0., 0., 0.], [0., 0., 0., 0., 0.],
                      [0., 0., 0., 0., 0.], [0., 0., 0., 0., 0.]])
    Qf = torch.tensor([[15., 0., 0., 0., 0.], [0., 15., 0., 0., 0.], [0., 0., 15., 0., 0.],
                       [0., 0., 0., 15., 0.], [0., 0., 0., 0., 15.]])
    R = 0.01 * torch.tensor([[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1]])
    T = 10
    attention_cost = 300.

    test_agent = SparseLQRAgent(A, B, Q, Qf, R, T, init_endogenous, attention_cost)
    opt_u = test_agent.get_actions()
    print(f"opt_u: {opt_u}")

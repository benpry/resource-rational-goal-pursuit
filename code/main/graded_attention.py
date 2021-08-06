"""
This file implements continuous attention. The main function is "analytic_attention" which is called by models that
use continuous attention.
"""
import torch
from Microworld_experiment import Microworld
import numpy as np
import itertools
from linear_quadratic_regulator import OptimalAgent


def compute_uaa(A, B, S, s, g, use_exo_cost):
    """
    Computes the uaa term for analytic attention
    """
    squared_dist_to_goal = (A.mv(s) - g).t().matmul(S.inverse()).dot(A.mv(s) - g)
    num1 = B.t().matmul(S.inverse().matmul(B)) * torch.sqrt(squared_dist_to_goal)
    num2 = S.inverse().mv(A.mv(s) - g).matmul(B).dot(S.inverse().mv(A.mv(s) - g).matmul(B)) / \
        (torch.sqrt(squared_dist_to_goal))

    ret = (num1 - num2) / squared_dist_to_goal

    # add the smallest possible diagonal matrix that makes the result invertible, for numerical stability
    for d in [1e-6, 1e-5, 1e-4, 1e-3, 1e-2]:
        if ret.det() == 0:
            ret += torch.eye(ret.shape[0]) * d
        else:
            break
    return ret


def compute_uax(A, B, S, s, g, is_in_B, loc, use_exo_cost):
    """
    Computes the uax term in the analytic attention
    """

    # We have different definitions of this term for elements in A and B
    if is_in_B:
        # configure dB to be all zeros except for a 1 in the position of loc
        dB = torch.zeros(B.shape, dtype=torch.float64)
        dB[loc] = 1
        num = S.inverse().mv(A.mv(s) - g).matmul(dB)
        denom = torch.sqrt((A.mv(s) - g).t().matmul(S.inverse()).dot(A.mv(s) - g))
        return num / denom
    else:
        dAs = torch.zeros(A.shape[0], dtype=torch.float64)
        dAs[loc[0]] = s[loc[1]]

        squared_dist_to_goal = (A.mv(s) - g).t().matmul(S.inverse()).dot(A.mv(s) - g)

        num1 = S.inverse().matmul(dAs).matmul(B) * torch.sqrt(squared_dist_to_goal)
        num2 = S.inverse().mv(A.mv(s) - g).dot(dAs) * (S.inverse().mv(A.mv(s) - g)).matmul(B) / \
            torch.sqrt(squared_dist_to_goal)

        return (num1 - num2) / squared_dist_to_goal


def reduce_microworld(microworld, m):
    """
    Apply attention matrix m to reduce the microworld

    microworld: the microworld to reduce
    m: the attention matrix
    """
    A_m = microworld.A * m[:, :microworld.A.shape[1]]
    B_m = microworld.B * m[:, microworld.A.shape[1]:]

    reduced_microworld = Microworld(A_m, B_m, init=microworld.endogenous_state, agent='hillclimbing')

    return reduced_microworld


def distance(s, g, scale):
    """
    Compute distance from state s to goal g
    """
    return torch.sqrt(torch.matmul(torch.matmul((s - g), scale).unsqueeze(0),
                                   (s - g).t()))


def analytic_attention(microworld, goal_scale, goal_state, t, attention_cost, step_size, clamp, use_exo_cost, exo_cost,
                       decision_type):
    """
    Computes the continuous attention and optimal action in the reduced micro-world analytically using a Taylor
    approximation.
    """
    A = microworld.A
    B = microworld.B
    s = microworld.endogenous_state

    # first compute the amount of attention to pay to each relationship
    uaa = compute_uaa(A, B, goal_scale, s, goal_state, use_exo_cost)
    A_attention = torch.zeros(A.shape)
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            loc = (i, j)
            if A[loc] == 0:
                A_attention[loc] = 0
                continue
            elif attention_cost == 0:
                A_attention[loc] = 1
                continue

            ax = compute_uax(A, B, goal_scale, s, goal_state, False, loc, use_exo_cost)
            cost_of_inattention = (A[loc] * ax.dot(uaa.inverse().mv(ax))).abs()
            if cost_of_inattention < 1e-20:
                attention = 0
            else:
                attention = max(1 - attention_cost / cost_of_inattention, 0)
            A_attention[loc] = attention

    B_attention = torch.zeros(B.shape)
    for i in range(B.shape[0]):
        for j in range(B.shape[1]):
            loc = (i, j)
            if B[loc] == 0:
                B_attention[loc] = 0
                continue
            elif attention_cost == 0:
                B_attention[loc] = 1
                continue

            ax = compute_uax(A, B, goal_scale, s, goal_state, True, loc, use_exo_cost)
            cost_of_inattention = (B[loc] * ax.dot(uaa.inverse().mv(ax))).abs()
            if cost_of_inattention < 1e-20:
                attention = 0
            else:
                attention = max(1 - attention_cost / cost_of_inattention, 0)
            B_attention[loc] = attention

    # Make sure that the diagonal of ones gets full attention
    for i in range(A.shape[0]):
        A_attention[i, i] = 1

    # create reduced simulated micro-worlds based on the attention values
    A_reduced = A_attention * A
    B_reduced = B_attention * B

    # compute the total amount of relationships not being attended to
    total_ignorance = len(torch.nonzero(A)) + len(torch.nonzero(B)) - A_attention.sum() - B_attention.sum()
    current_goal_dist = distance(s, goal_state, goal_scale)

    # create a new micro-world with the attention-reduced matrices
    reduced_microworld = Microworld(A_reduced, B_reduced, init=s, agent='hillclimbing')

    # this is for backwards-compatibility with an older version of the task. use_exo_cost should be true for the
    # most recent version.
    if use_exo_cost:

        # compute the gradient
        gradient = -torch.matmul(torch.div(
            torch.matmul(goal_scale, (A_reduced.mv(s) - goal_state)), current_goal_dist),
            B_reduced)

        # Implementations of different possible decision functions. We ultimately decided to only use the gradient
        # with optimal step size, which is implemented in the "else" part of this if statement.
        if decision_type == 'least_squares':
            direction = np.linalg.lstsq(B_reduced, goal_state - A_reduced.mv(s), rcond=None)
            direction = torch.tensor(direction[0])

            if direction.norm() == 0:
                opt_step_size = 0
            else:
                opt_step_size = - goal_scale.inverse().matmul(A_reduced.mv(s) - goal_state).dot(B_reduced.mv(direction))\
                                / (B_reduced.mv(direction).matmul(goal_scale.inverse()).dot(B_reduced.mv(direction))
                                   + exo_cost * (direction.dot(direction)))

            exogenous = step_size * opt_step_size * direction

            reduced_microworld.step(exogenous)

            loss = distance(reduced_microworld.endogenous_state, goal_state, goal_scale)**2 \
                + attention_cost * (torch.sum(A_reduced) - A.shape[0] + torch.sum(B_reduced)) + \
                exo_cost * exogenous.dot(exogenous)

        elif decision_type == 'per_variable':
            variable_order = s.abs().numpy().argsort()
            exogenous = torch.zeros(B.shape[1])

            for _ in range(len(s)):
                i = np.argmin(variable_order)
                variable_order[i] = len(s)

                # get the index of the corresponding exogenous variable
                exo_i = B_reduced[i, :].abs().argmax()
                direction = torch.zeros(B.shape[1])
                direction[exo_i] = 1.

                var_size = -(A_reduced.mv(s) + B_reduced.mv(exogenous)).dot(B_reduced.mv(direction)) /\
                    (B_reduced.mv(direction).dot(B_reduced.mv(direction)) + exo_cost)

                exogenous[exo_i] += var_size

            exogenous *= step_size

            loss = distance(reduced_microworld.endogenous_state, goal_state, goal_scale) ** 2 \
                + attention_cost * (torch.sum(A_reduced) - A.shape[0] + torch.sum(B_reduced)) + \
                exo_cost * exogenous.dot(exogenous)

        elif decision_type == 'one_step_lqr':
            Q = torch.tensor([[0., 0., 0., 0., 0.], [0., 0., 0., 0., 0.], [0., 0., 0., 0., 0.],
                              [0., 0., 0., 0., 0.], [0., 0., 0., 0., 0.]])
            Qf = torch.tensor([[1., 0., 0., 0., 0.], [0., 1., 0., 0., 0.], [0., 0., 1., 0., 0.],
                               [0., 0., 0., 1., 0.], [0., 0., 0., 0., 1.]])
            R = torch.tensor([[0.01, 0., 0., 0.], [0., 0.01, 0., 0.], [0., 0., 0.01, 0.], [0., 0., 0., 0.01]])

            opt_agent = OptimalAgent(A_reduced, B_reduced, Q, Qf, R, 1, s)
            opt_action = opt_agent.get_actions()[0]
            exogenous = opt_action * step_size

            reduced_microworld.step(exogenous)

            loss = distance(reduced_microworld.endogenous_state, goal_state, goal_scale)**2 \
                + attention_cost * (torch.sum(A_reduced) - A.shape[0] + torch.sum(B_reduced)) + \
                exo_cost * exogenous.dot(exogenous)
        else:
            # compute the optimal step size to take
            if gradient.norm() == 0:
                opt_step_size = 0
            else:
                opt_step_size = - goal_scale.inverse().matmul(A_reduced.mv(s) - goal_state).dot(B_reduced.mv(gradient))\
                                / (B_reduced.mv(gradient).matmul(goal_scale.inverse()).dot(B_reduced.mv(gradient))
                                   + exo_cost * (gradient.dot(gradient)))

            exogenous = step_size * opt_step_size * gradient

            reduced_microworld.step(exogenous)

            loss = distance(reduced_microworld.endogenous_state, goal_state, goal_scale)**2 \
                + attention_cost * (torch.sum(A_reduced) - A.shape[0] + torch.sum(B_reduced)) + \
                exo_cost * exogenous.dot(exogenous)
    else:
        # this code is from an older version of the task
        exogenous = step_size * gradient
        if exogenous.abs().sum() > clamp:
            exogenous = (clamp / exogenous.abs().sum()) * exogenous

        reduced_microworld.step(exogenous)

        loss = distance(reduced_microworld.endogenous_state, goal_state, goal_scale) \
            + attention_cost * (torch.sum(A_reduced) - A.shape[0] + torch.sum(B_reduced))

    return (exogenous, loss, total_ignorance)

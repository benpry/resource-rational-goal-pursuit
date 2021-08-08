"""
This file implements continuous attention. The main function is "analytic_attention" which is called by models that
use continuous attention.
"""
import torch
from Microworld_experiment import Microworld

def compute_uaa(A, B, S, s, g):
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


def compute_uax(A, B, S, s, g, is_in_B, loc):
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

def distance(s, g, scale):
    """
    Compute distance from state s to goal g
    """
    return torch.sqrt(torch.matmul(torch.matmul((s - g), scale).unsqueeze(0),
                                   (s - g).t()))


def analytic_attention(microworld, goal_scale, goal_state, attention_cost, step_size, exo_cost):
    """
    Computes the continuous attention and optimal action in the reduced micro-world analytically using a Taylor
    approximation.
    """
    A = microworld.A
    B = microworld.B
    s = microworld.endogenous_state

    # first compute the amount of attention to pay to each relationship
    uaa = compute_uaa(A, B, goal_scale, s, goal_state)
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

            ax = compute_uax(A, B, goal_scale, s, goal_state, False, loc)
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

            ax = compute_uax(A, B, goal_scale, s, goal_state, True, loc)
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
    reduced_microworld = Microworld(A_reduced, B_reduced, init=s)

    # compute the gradient
    gradient = -torch.matmul(torch.div(
        torch.matmul(goal_scale, (A_reduced.mv(s) - goal_state)), current_goal_dist),
        B_reduced)

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

    return (exogenous, loss, total_ignorance)

"""
This file implements continuous attention. The main function is "analytic_attention" which is called by models that
use continuous attention.
"""
import torch

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
        # dAs should be the product of dA (all zeros but a 1 in the location of x) and s, the current state
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
    # compute d^2C/da^2
    uaa = compute_uaa(A, B, goal_scale, s, goal_state)
    A_attention = torch.zeros(A.shape, dtype=torch.float64)
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            loc = (i, j)
            # don't pay attention to non-existent connections
            if A[loc] == 0:
                A_attention[loc] = 0
                continue
            # pay full attention if attention is free
            elif attention_cost == 0:
                A_attention[loc] = 1
                continue

            # compute da/dm _i
            ax = compute_uax(A, B, goal_scale, s, goal_state, False, loc)
            # combine the derivatives to get the cost of inattention
            cost_of_inattention = (A[loc] * ax.dot(uaa.inverse().mv(ax))).abs()
            if cost_of_inattention < 1e-20:
                attention = 0
            else:
                # compute the amount of attention to pay
                attention = max(1 - attention_cost / cost_of_inattention, 0)
            A_attention[loc] = attention

    B_attention = torch.zeros(B.shape, dtype=torch.float64)
    for i in range(B.shape[0]):
        for j in range(B.shape[1]):
            loc = (i, j)
            # don't pay attention to non-existent connections
            if B[loc] == 0:
                B_attention[loc] = 0
                continue
            # pay full attention if attention is free
            elif attention_cost == 0:
                B_attention[loc] = 1
                continue

            # compute dC/dx
            ax = compute_uax(A, B, goal_scale, s, goal_state, True, loc)
            # combine the derivatives to get the cost of inattention
            cost_of_inattention = (B[loc] * ax.dot(uaa.inverse().mv(ax))).abs()
            # compute the amount of attention to pay
            if cost_of_inattention < 1e-20:
                attention = 0
            else:
                attention = max(1 - attention_cost / cost_of_inattention, 0)
            B_attention[loc] = attention

    # Make sure that the diagonal of ones gets full attention
    for i in range(A.shape[0]):
        A_attention[i, i] = 1

    # create a reduced simulated micro-world based on the attention values
    A_reduced = A_attention * A
    B_reduced = B_attention * B

    # compute the total amount of relationship being ignored
    total_ignorance = len(torch.nonzero(A)) + len(torch.nonzero(B)) - A_attention.sum() - B_attention.sum()
    total_attention = A_attention.sum() + B_attention.sum() - 5
    current_goal_dist = distance(A_reduced.mv(s), goal_state, goal_scale)

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

    microworld.step(exogenous)

    loss = torch.sqrt(distance(microworld.endogenous_state, goal_state, goal_scale)**2 +
                      exo_cost * exogenous.dot(exogenous)) + attention_cost * total_attention
    microworld.endogenous_state = s

    return (exogenous, loss, total_ignorance)

"""
Algorithms to calculate endpoint-conditioned Markov chains
"""
# -*- coding: utf-8 -*-
# =============================================================================
# File      : endpoint_conditioned_markov_chains.py
# Author    : Juergen Hackl <hackl@ibi.baug.ethz.ch>
# Creation  : 2019-02-22
# Time-stamp: <Fre 2019-02-22 09:58 juergen>
#
# Copyright (c) 2019 Juergen Hackl <hackl@ibi.baug.ethz.ch>
# =============================================================================

import numpy as _np


def number_of_visits(network, start_node, end_node, visit_node, n):
    """Expected number of visits in state c.

    The endpoint-conditioned mean value of the number of visits $T_c$ to a state
    $c$, given the the starting state $X_0 = a$ and ending state $X_n = b$ of a
    Markov process is given by:

    .. math: :
    \E[T_c | X_0 = a, X_n = b] &= \frac{\sum_{n = 0} ^\infty M_c(n, a, b)
                                       \P(n, t)}{\P(X_n=b | X_0=a)}\\

    Parameters:
    -----------
    network: Network, TemporalNetwork, HigherOrderNetwork
        The temporal, first-order, or higher-order network, which
        will be used to randomly generate a walk through a network.

    start_node: str (a)
        The(higher-order) node in which the random walk will be started.

    end_node: str (b)
        The(higher-order) node in which the random walk will end.

    visit_node: str (c)
        The(higher-order) node for which the expected number of visits is
        calculated.

    n: int
        The length of the walk to be generated.

    Returns:
    --------
    visits: float (T_c)
        The expected number of visits in state c.

    """
    # get the transion matrix for the network
    P = network.transition_matrix().todense().transpose()

    # map the node names to theire indices
    idx_map = network.node_to_name_map()

    # initialize variables
    a = idx_map[start_node]
    b = idx_map[end_node]
    c = idx_map[visit_node]
    M = 0

    # NOTE: for lager networks this might be become computational
    # intensive.
    # TODO: Find a better solution!
    for i in range(n):
        M += _np.linalg.matrix_power(P, i)[a, c] * \
            _np.linalg.matrix_power(P, n-i)[c, b]

    return M/_np.linalg.matrix_power(P, n)[a, b]


def number_of_transitions(network, start_node, end_node, from_node, to_node, n):
    """Expected number of transitions from c to d.

    The endpoint-conditioned mean value of the number of state changes $N_{c,d}
    from state $c$ to state $d$, given the the starting state $X_0 = a$ and
    ending state $X_n = b$ of a Markov process is given by:

    .. math: :
        \E[N_{c,d}|X_0=a,X_n=b] &= \frac{\sum_{n=0}^\infty M_{c,d}(n,a,b)
        \P(n,t)}{\P(X_n=b|X_0=a)}

    Parameters:
    -----------
    network: Network, TemporalNetwork, HigherOrderNetwork
        The temporal, first-order, or higher-order network, which
        will be used to randomly generate a walk through a network.

    start_node: str (a)
        The(higher-order) node in which the random walk will be started.

    end_node: str (b)
        The(higher-order) node in which the random walk will end.

    from_node: str (c)
        The(higher-order) node from which the expected number of transitions is
        calculated.

    to_node: str (d)
        The(higher-order) node to which the expected number of transitions is
        calculated.

    n: int
        The length of the walk to be generated.

    Returns:
    --------
    transitions: float (N_{c,d})
        The expected number of transitions from c to d.

    """
    # get the transion matrix for the network
    P = network.transition_matrix().todense().transpose()

    # map the node names to theire indices
    idx_map = network.node_to_name_map()

    # initialize variables
    a = idx_map[start_node]
    b = idx_map[end_node]
    c = idx_map[from_node]
    d = idx_map[to_node]
    M = 0

    # NOTE: for lager networks this might be become computational
    # intensive.
    # TODO: Find a better solution!
    for i in range(1, n+1):
        M += _np.linalg.matrix_power(P, i-1)[a, c] * \
            P[c, d] * _np.linalg.matrix_power(P, n-i)[d, b]

    return M/_np.linalg.matrix_power(P, n)[a, b]


def distribution_of_transitions(network, start_node, end_node, from_node, to_node, n, k):
    """Distribution for transitions from c to d.

    The endpoint-conditioned distribution of the number of state changes $N_{c,d}
    from state $c$ to state $d$, given the the starting state $X_0 = a$ and
    ending state $X_n = b$ of a Markov process is given by:

    .. math::

        \P(N_{c,d}=k|X_0=a,X_n=b) =
        \frac{\P_{c,d}(k,n,a,b)}{\P(X_n=b|X_0=a)}=\frac{\QM_{c,d}(k,n)}{(\TM^n)_{a,b}}

    Parameters:
    -----------
    network: Network, TemporalNetwork, HigherOrderNetwork
        The temporal, first-order, or higher-order network, which
        will be used to randomly generate a walk through a network.

    start_node: str (a)
        The(higher-order) node in which the random walk will be started.

    end_node: str (b)
        The(higher-order) node in which the random walk will end.

    from_node: str (c)
        The(higher-order) node from which the expected number of transitions is
        calculated.

    to_node: str (d)
        The(higher-order) node to which the expected number of transitions is
        calculated.

    n: int
        The length of the walk to be generated.

    k: int
        The number of times an event occurs in an interval

    Returns:
    --------
    probability: float
        Returns the probability that a transition from c to d is observed
        exactly k times.

    """
    # get the transion matrix for the network
    P = network.transition_matrix().todense().transpose()

    # map the node names to theire indices
    idx_map = network.node_to_name_map()

    # initialize variables
    a = idx_map[start_node]
    b = idx_map[end_node]
    c = idx_map[from_node]
    d = idx_map[to_node]

    n_states = len(P)
    U = _np.zeros((n_states, n_states))
    U[c, d] = 1

    Pcd = _np.zeros((n+1, n+1, n_states, n_states))
    Pcd[0, 0] = _np.eye(n_states)
    Pcd[0, 1] = P - P * U
    Pcd[1, 1] = P * U

    for i in range(2, n+1):
        Pcd[0, i] = Pcd[0, i-1].dot(P-P * U)

    for i in range(1, n+1):
        for j in range(1, i):
            Pcd[j, i] = Pcd[j-1, i-1].dot(P*U) + Pcd[j, i-1].dot(P-P*U)

    probability = Pcd[k, n, a, b]/_np.linalg.matrix_power(P, n)[a, b]

    # filter out numerical erros (i.e. negativ values)
    if probability >= 0:
        return probability
    else:
        return 0.0


def distribution_of_visits(network, start_node, end_node, visit_node, n, k):
    """Distribution of visits in state c.

    See also
    --------
    `distribution_of_transitions`

    """
    return distribution_of_transitions(network, start_node, end_node,
                                       from_node=visit_node,
                                       to_node=visit_node, n=n, k=k)


def transition_probability(network, end_node, from_node, to_node, n, i):
    """Endpoint-conditioned transition probability between c and d.

    Given the starting state $X_0=a$ and ending state $X_n=b$ of a Markov
    process, the transition probabilities for any state $X_i$ in between
    (i.e. $i=1,\dots,n-1$) are given by \cite{Hobolth2009}:

    .. math::
        \P(X_i=v|X_{i-1} = u,X_n=b) = \TM_{u,v}^{a,b} =
        \frac{\TM_{u,v}(\TM^{n-i})_{v,b}}{(\TM^{n-i+1})_{u,b}}

    Parameters:
    -----------
    network: Network, TemporalNetwork, HigherOrderNetwork
        The temporal, first-order, or higher-order network, which
        will be used to randomly generate a walk through a network.

    end_node: str (b)
        The(higher-order) node in which the random walk will end.

    from_node: str (c)
        The(higher-order) node from which the expected number of transitions is
        calculated.

    to_node: str (d)
        The(higher-order) node to which the expected number of transitions is
        calculated.

    n: int
        The length of the walk to be generated.

    i: int
        Already walked steps.

    Returns:
    --------
    probability: float
        Returns the probability for a transition from c to d at step i out of n
        steps.

    """
    # get the transion matrix for the network
    P = network.transition_matrix().todense().transpose()

    # map the node names to theire indices
    idx_map = network.node_to_name_map()

    # initialize variables
    b = idx_map[end_node]
    c = idx_map[from_node]
    d = idx_map[to_node]

    return P[c, d] * _np.linalg.matrix_power(P, n-i)[d, b] / _np.linalg.matrix_power(P, n-i+1)[c, b]


def path_probability(network, end_node, n, path):
    """Probability of observing path p."""

    # TODO: Make it to more general case,
    # i.e. part of a path, different starting times, ...
    # initialize variables
    _temp_path = []

    for i in range(1, n+1):
        p = transition_probability(network, end_node, path[i-1], path[i], n, i)
        _temp_path.append(p)

    # print(_temp_path)
    return _np.prod(_temp_path)


def generate_path(network, start_node, end_node, n):
    """Genarate a random path between a and b with length n.

    """
    # get the transion matrix for the network
    P = network.transition_matrix().todense().transpose()

    # map the node names to theire indices
    idx_map = network.node_to_name_map()
    inv_map = {v: k for k, v in idx_map.items()}

    # initialize variables
    a = idx_map[start_node]
    b = idx_map[end_node]

    # Get the number of states and initialize vector of states
    n_states = len(P)

    # Initialize number of jumps and conditional probability of n jumps
    n_jumps = n

    # Initialize a 3rd order tensors for storing powers of P
    P_pow = _np.zeros((n_states, n_states, n_jumps+1))
    for i in range(n_jumps+1):
        P_pow[:, :, i] = _np.linalg.matrix_power(P, i)

    # initialize the path matrix
    path_nrows = n_jumps + 1
    path = _np.zeros((path_nrows, 2))
    path[0, 0] = 0
    path[0, 1] = a

    # transition times are uniformly distributed in the
    # interval. Sample them, sort them, and place in path.
    transitions = range(1, n_jumps+1)
    path[1:n_jumps+1, 0] = transitions

    # Sample the states at the transition times
    for j in range(1, n_jumps+1):
        Px = _np.asarray(P[int(path[j-1, 1])])[0]
        Pn1 = P_pow[:, b, n_jumps-j]
        Pn2 = P_pow[int(path[j-1, 1]), b, n_jumps-j+1]
        state_probs = Px*Pn1/Pn2

        path[j, 1] = _np.random.choice(
            n_states, 1, False, state_probs)[0]

    # Determine which transitions are virtual transitions
    # keep_inds = np.ones(path_nrows, dtype=bool)
    # for j in range(1, n_jumps + 1):
    #     if path[j, 1] == path[j-1, 1]:
    #         keep_inds[j] = False

    # create a matrix for the complete path without virtual jumps
    # return path[keep_inds]
    return [inv_map[int(i)] for i in path.T[1]]


# =============================================================================
# eof
#
# Local Variables:
# mode: python
# mode: linum
# mode: auto-fill
# fill-column: 80
# End:

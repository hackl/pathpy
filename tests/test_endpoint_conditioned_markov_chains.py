#!/usr/bin/python -tt
# -*- coding: utf-8 -*-
# =============================================================================
# File      : test_endpoint_conditioned_markov_chains.py
# Author    : Juergen Hackl <hackl@ibi.baug.ethz.ch>
# Creation  : 2019-02-22
# Time-stamp: <Fre 2019-02-22 14:53 juergen>
#
# Copyright (c) 2019 Juergen Hackl <hackl@ibi.baug.ethz.ch>
# =============================================================================

import pytest
import pathpy as pp
from pathpy.algorithms import endpoint_conditioned_markov_chains as ecmc


@pytest.fixture
def net():
    net = pp.Network()
    net.add_node('a')
    net.add_node('b')
    net.add_node('c')
    net.add_node('d')
    net.add_node('e')
    net.add_node('f')
    net.add_node('g')

    net.add_edge('a', 'b')
    net.add_edge('a', 'c')
    net.add_edge('b', 'c')
    net.add_edge('b', 'd')
    net.add_edge('d', 'e')
    net.add_edge('d', 'f')
    net.add_edge('d', 'g')
    net.add_edge('e', 'f')
    net.add_edge('f', 'g')
    return net


def test_number_of_visits(net):
    assert ecmc.number_of_visits(net, 'a', 'g', 'b', 3) == 1.0


def test_number_of_transitions(net):
    assert ecmc.number_of_transitions(net, 'a', 'g', 'b', 'd', 3) == 1.0


def test_distribution_of_transitions(net):
    assert ecmc.distribution_of_transitions(
        net, 'a', 'g', 'b', 'd', 3, 1) == 1.0


def test_distribution_of_visits(net):
    assert ecmc.distribution_of_visits(net, 'a', 'g', 'b', 3, 1) == 1.0


def test_transition_probability(net):
    assert ecmc.transition_probability(net, 'g', 'f', 'g', 1, 1) == 1.0


def test_path_probability(net):
    assert ecmc.path_probability(net, 'g', 3, ['a', 'b', 'd', 'g']) == 1.0


def test_generate_path(net):
    assert ecmc.generate_path(net, 'a', 'g', 3) == ['a', 'b', 'd', 'g']


# =============================================================================
# eof
#
# Local Variables:
# mode: python
# mode: linum
# mode: auto-fill
# fill-column: 80
# End:

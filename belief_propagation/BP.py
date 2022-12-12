# -*- coding: utf-8 -*-
# @Author: Sadamori Kojaku
# @Date:   2022-11-02 13:34:28
# @Last Modified by:   Sadamori Kojaku
# @Last Modified time: 2022-12-11 21:24:57
import os
import pathlib
import tempfile
import numpy as np
import pandas as pd
from scipy import sparse
import networkx as nx


def detect(
    A, q, iters=1, p_init=None, cab_init=None, ave_deg=None, params_sbm="", mute=True
):
    """Belief propagation

    :param A: Adjacency matrix
    :type A: sparse.csr_matrix
    :param q: Number of communities
    :type q: int
    :param iters: Number of communication detections to run, defaults to 1. The best communities in terms of the free energy will be returned.
    :type iters: int, optional
    :param params_sbm: parameters to be passed to the original "sbm" program, defaults to "". See the author's code for the details.
    :type params_sbm: str, optional
    :param mute: mute = True to mute the verbose, defaults to True
    :type mute: bool, optional
    :return: numpy.ndarray
    :rtype: an array of community memberships
    """

    if ave_deg is None:
        n_nodes = A.shape[0]
        ave_deg = len(A.data) / n_nodes

    if p_init is None:
        p_init = np.ones(q) / q

    if cab_init is None:
        cab_init = np.ones((q, q)) + np.eye(q) * (q * ave_deg - 1)

    params_sbm += " -p" + ",".join(["%f" % d for d in p_init])
    params_sbm += " -c" + ",".join(["%d" % d for d in cab_init.reshape(-1)])

    A.setdiag(0)
    A.eliminate_zeros()
    A.data = A.data * 0 + 1
    A = sparse.csr_matrix(A)
    G = nx.from_scipy_sparse_array(A)

    for (n1, n2, d) in G.edges(data=True):
        d.clear()
    for (n1, d) in G.nodes(data=True):
        d.clear()
    with tempfile.TemporaryDirectory() as tmpdirname:
        root = pathlib.Path(__file__).parent.absolute()
        # graph_file_name = f"./graph.gml"
        # output_file_name = f"./tmp.csv"
        graph_file_name = f"{tmpdirname}/graph.gml"
        output_file_name = f"{tmpdirname}/tmp.csv"

        nx.write_gml(G, graph_file_name)

        if mute:
            mute = ">/dev/null"
        else:
            mute = ""

        energy = np.inf
        cids = np.zeros(A.shape[0], dtype=int)
        for _ in range(iters):
            os.system(
                # f"{root}/sbm spc -q {q} -l {graph_file_name} -w {output_file_name} {params_sbm} >/dev/null"
                f"{root}/sbm learn -q {q} -l {graph_file_name} -w {output_file_name} {params_sbm} {mute}"
                # f"{root}/sbm learn -q {q} -l {graph_file_name} -w {output_file_name} {params_sbm} >/dev/null"
            )
            df = pd.read_csv(output_file_name)
            energy_t = df["energy"].values[0]
            if energy > energy_t:
                energy = energy_t
                cids = df["block"].values
                cids = np.unique(cids, return_inverse=True)[1]

    return cids

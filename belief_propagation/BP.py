# -*- coding: utf-8 -*-
# @Author: Sadamori Kojaku
# @Date:   2022-11-02 13:34:28
# @Last Modified by:   Sadamori Kojaku
# @Last Modified time: 2022-12-12 06:34:45
import os
import pathlib
import tempfile
import numpy as np
import pandas as pd
from scipy import sparse
import networkx as nx


def detect(
    A,
    q,
    iters=1,
    init_memberships=None,
    params_sbm="",
    mute=True,
):
    """Belief propagation

    :param A: Adjacency matrix
    :type A: sparse.csr_matrix
    :param q: Number of communities
    :type q: int
    :param iters: Number of communication detections to run, defaults to 1. The best communities in terms of the free energy will be returned.
    :type iters: int, optional
    :param init_memberships: p_init and cab will be initialized by using the memberships.
    :type init_memberships: numpy array or list.
    :param p_init: p_init[i] is the fraction of the ith community
    :type p_init: numpy array or list.
    :param params_sbm: parameters to be passed to the original "sbm" program, defaults to "". See the author's code for the details.
    :type params_sbm: str, optional
    :param mute: mute = True to mute the verbose, defaults to True
    :type mute: bool, optional
    :return: numpy.ndarray
    :rtype: an array of community memberships
    """
    if init_memberships is not None:
        n_nodes = A.shape[0]
        l, memberships = np.unique(init_memberships, return_inverse=True)
        K = len(l)
        U = sparse.csr_matrix(
            (np.ones_like(memberships), (np.arange(n_nodes), memberships)),
            shape=(n_nodes, K),
        )
        Nc = np.array(U.sum(axis=0)).reshape(-1)
        cab_init = np.maximum(n_nodes * ((U.T @ A @ U).toarray() / np.outer(Nc, Nc)), 1)
        p_init = Nc / n_nodes
    else:
        cab_init, p_init = None, None

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
        # cab_file_name = f"./tmp.cab"
        # output_file_name = f"./tmp.csv"
        graph_file_name = f"{tmpdirname}/graph.gml"
        cab_file_name = f"{tmpdirname}/tmp.cab"
        output_file_name = f"{tmpdirname}/tmp.csv"

        nx.write_gml(G, graph_file_name)
        if cab_init is not None:
            generate_cab_file(cab_file_name, p_init, cab_init)
            params_sbm += f" -L {cab_file_name}"

        if mute:
            mute = ">/dev/null"
        else:
            mute = ""

        energy = np.inf
        cids = np.zeros(A.shape[0], dtype=int)
        for _ in range(iters):
            os.system(
                f"{root}/sbm learn -q {q} -l {graph_file_name} -w {output_file_name} {params_sbm} {mute}"
            )
            df = pd.read_csv(output_file_name)
            energy_t = df["energy"].values[0]
            if energy > energy_t:
                energy = energy_t
                cids = df["block"].values
                cids = np.unique(cids, return_inverse=True)[1]

    return cids


def generate_cab_file(filename, p, cab):
    pstr = " ".join([f"{d:.4f}" for d in p])
    cabstr = "\n".join(
        [
            " ".join([f"{cab[r,c]:.4f}" for c in range(cab.shape[1])])
            for r in range(cab.shape[0])
        ]
    )
    s = f"#Vector_na:\n{pstr}\n#Matrix_cab:\n{cabstr}"
    text_file = open(filename, "w")
    text_file.write(s)
    text_file.close()
    return

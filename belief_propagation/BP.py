# -*- coding: utf-8 -*-
# @Author: Sadamori Kojaku
# @Date:   2022-11-02 13:34:28
# @Last Modified by:   Sadamori Kojaku
# @Last Modified time: 2022-11-11 06:39:03
import os
import pathlib
import tempfile
import numpy as np
import pandas as pd
from scipy import sparse
import networkx as nx


def detect(A, q, params_sbm="", mute=True):
    A.setdiag(0)
    A.eliminate_zeros()
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
        os.system(
            # f"{root}/sbm spc -q {q} -l {graph_file_name} -w {output_file_name} {params_sbm} >/dev/null"
            f"{root}/sbm learn -q {q} -l {graph_file_name} -w {output_file_name} {params_sbm} {mute}"
            # f"{root}/sbm learn -q {q} -l {graph_file_name} -w {output_file_name} {params_sbm} >/dev/null"
        )
        df = pd.read_csv(output_file_name)
        cids = df["block"].values
        cids = np.unique(cids, return_inverse=True)[1]

    return cids, df

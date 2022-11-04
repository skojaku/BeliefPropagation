# -*- coding: utf-8 -*-
# @Author: Sadamori Kojaku
# @Date:   2022-11-02 13:34:28
# @Last Modified by:   Sadamori Kojaku
# @Last Modified time: 2022-11-03 21:19:08
import os
import pathlib
import tempfile
import numpy as np
import pandas as pd
from scipy import sparse
import networkx as nx

def detect(A, q):
    G = nx.from_scipy_sparse_array(A)
    for (n1, n2, d) in G.edges(data=True):
        d.clear()
    with tempfile.TemporaryDirectory() as tmpdirname:
        root = pathlib.Path(__file__).parent.absolute()
        graph_file_name = f"{tmpdirname}/graph.gml"
        output_file_name = f"{tmpdirname}/tmp.csv"
        nx.write_gml(G, graph_file_name)
        os.chdir(tmpdirname)
        os.system(
            f"{root}/sbm learn -q {q} -l {graph_file_name} -w {output_file_name}"
            #f"{root}/sbm learn -q {q} -l {graph_file_name} -w {output_file_name} >/dev/null"
        )
        cids = pd.read_csv(output_file_name)["block"].values
        cids = np.unique(cids, return_inverse=True)[1]
    return cids

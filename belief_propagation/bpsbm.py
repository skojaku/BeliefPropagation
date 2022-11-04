# -*- coding: utf-8 -*-
# @Author: Sadamori Kojaku
# @Date:   2022-11-02 13:34:28
# @Last Modified by:   Sadamori Kojaku
# @Last Modified time: 2022-11-03 21:19:08
# %%
import os
import pathlib
import tempfile
import numpy as np
import pandas as pd
from scipy import sparse
import networkx as nx


class BeliefPropagation:
    def __init__(self, q):
        self.q = q

    def init_seed(self, path, seed):
        """Set the seed file for the external program."""
        with open("{path}/time_seed.dat".format(path=path), "w", encoding="utf-8") as f:
            f.write("%d" % seed)
            f.close()

    def detect(self, A):
        G = nx.from_scipy_sparse_array(A)
        for (n1, n2, d) in G.edges(data=True):
            d.clear()
        with tempfile.TemporaryDirectory() as tmpdirname:
            root = pathlib.Path(__file__).parent.absolute()
            graph_file_name = f"{tmpdirname}/graph.gml"
            output_file_name = f"{tmpdirname}/tmp.csv"
            nx.write_gml(G, graph_file_name)
            os.chdir(tmpdirname)
            q = self.q
            os.system(
                f"{root}/sbm learn -q {q} -l {graph_file_name} -w {output_file_name} >/dev/null"
            )
            cids = pd.read_csv(output_file_name)["group"].values

            cids = np.unique(cids, return_inverse=True)[1]

        return cids


G = nx.karate_club_graph()
G = nx.convert_node_labels_to_integers(G)
A = nx.to_scipy_sparse_matrix(G)
A.data = A.data.astype(float) * 0 + 1
# import igraph

# sources, targets = A.nonzero()
# edgelist = zip(sources.tolist(), targets.tolist())
# g = igraph.Graph(edgelist)
# g.write_gml("tmp.gml")
# nx.write_gml(G, "tmp.gml")
# G = nx.read_gml("tmp.gml")
model = BeliefPropagation(q=2)
cids = model.detect(A)

# %%
G = nx.from_scipy_sparse_matrix(A, create_using=nx.Graph)
nx.write_gml(G, "temp.gml")
G = nx.read_gml("temp.gml")
G = nx.convert_node_labels_to_integers(G)
A = nx.adjacency_matrix(G)

import belief_propagation
import networkx as nx

# Create a karate club network.
A = nx.to_scipy_sparse_matrix(nx.karate_club_graph()) 

# Detect communities
# A: scipy sparse matrix. CSR format. 
# q: Number of communities
community_ids = belief_propagation.detect(A, q=3) 
print(community_ids)

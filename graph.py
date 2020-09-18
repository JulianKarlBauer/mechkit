import mechkit
import numpy as np
import networkx as nx

np.set_printoptions(
    linewidth=140,
    precision=4,
    # suppress=False,
)

graph = nx.DiGraph()

# nodes = [f"not_{i}" for i in range(10)]
# g.add_nodes_from(nodes)


def xxs_to_s(x):
    return x + 1


def s_to_m(x):
    return x + 2


def m_to_l(x):
    return x + 3


edges = [
    ("xxs", "s", dict(func=xxs_to_s)),
    ("s", "m", dict(func=s_to_m)),
    ("m", "l", dict(func=m_to_l)),
    ("l", "xl", dict(func=lambda x: x + 4)),
]

graph.add_edges_from(edges)

# path = nx.dijkstra_path(G=graph, source="xxs", target="l")

path = nx.shortest_path(G=graph, source="xxs", target="l")

steps = list(nx.utils.pairwise(path))

x = 10
for step_start, step_end in steps:
    func = graph.edges[step_start, step_end]["func"]
    x = func(x)
    print(func.__name__)
    print(x)

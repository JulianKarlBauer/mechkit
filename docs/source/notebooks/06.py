# # Notation: Draw Supported Notations of Explicit Converter

import mechkit
import networkx as nx
import matplotlib.pyplot as plt

plot_options = dict(
    node_color="yellow",
    node_size=2000,
    width=2,
    arrows=True,
    font_size=10,
    font_color="black",
)

converter = mechkit.notation.ExplicitConverter()

for entity_type, graph in converter.graphs_dict.items():

    pos = nx.spring_layout(graph, seed=1)

    fig = plt.figure()
    nx.draw_networkx(graph, **plot_options)
    plt.gca().set_title(entity_type)
    plt.tight_layout()

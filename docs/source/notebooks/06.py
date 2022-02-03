# # Draw Supported Notations of Explicit Converter

import mechkit
import networkx as nx
import matplotlib.pyplot as plt
import os


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
    # nx.draw_networkx_nodes(graph, pos, node_color="yellow", width=10)
    # nx.draw_networkx_edges(graph, pos)
    # nx.draw_networkx_labels(graph, pos, font_size=10, font_color="black")

    plt.gca().set_title(entity_type)
    plt.tight_layout()

    path_picture = os.path.join(entity_type + ".png")
    plt.savefig(path_picture, dpi=300)

    plt.close(fig)
    # plt.show()

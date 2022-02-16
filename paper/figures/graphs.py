# # Notation: Draw Supported Notations of Explicit Converter

import mechkit
import matplotlib.pyplot as plt
import netgraph

plot_options = dict(
    edge_width=2,
    arrows=True,
    # node_shape="h",
    node_size=10,
    node_color="#E69F00",
    node_edge_width=0,
    node_labels=True,
    node_label_fontdict=dict(size=30),
)

pos = {
    "mandel6": (0.33, 0.5),
    "mandel9": (0.15, 0.25),
    "tensor": (0.15, 0.75),
    "voigt": (0.67, 0.5),
    "umat": (0.85, 0.25),
    "vumat": (0.85, 0.75),
    "abaqusMaterialAnisotropic": (1.0, 0.5),
}
converter = mechkit.notation.ExplicitConverter()

for entity_type, graph in converter.graphs_dict.items():
    fig = plt.figure(figsize=(15, 15))
    g = netgraph.Graph(graph, node_layout=pos, **plot_options)
    if entity_type == "stiffness":
        g.node_label_artists["abaqusMaterialAnisotropic"].set_size(13),
    plt.gca().set_title(entity_type)
    plt.tight_layout()

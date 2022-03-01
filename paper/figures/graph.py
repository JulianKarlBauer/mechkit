# # Notation: Draw Supported Notations of Explicit Converter

import mechkit
import matplotlib.pyplot as plt
import netgraph

plot_options = dict(
    edge_color="black",
    edge_alpha=1.0,
    edge_width=1.33,
    arrows=True,
    # node_shape="h",
    node_size=10,
    node_color="#E69F00",
    node_edge_width=0,
    node_labels=True,
    node_label_fontdict=dict(size=30),
)

titlefont_options = dict(
    fontsize=32,
)
alias_abaquas_material_anisotropic = "abaqMatAniso" # "abaqusMatAniso"
pos = {
    "mandel6": (0.33, 0.5),
    "mandel9": (0.15, 0.25),
    "tensor": (0.15, 0.75),
    "voigt": (0.67, 0.5),
    "umat": (0.85, 0.25),
    "vumat": (0.85, 0.75),
    alias_abaquas_material_anisotropic: (1.0, 0.5),
}
converter = mechkit.notation.ExplicitConverter()

# edges = [(edge[0], edge[1]) for edge in converter.edges_dict["stiffness"]]
edges = [
    ("tensor", "mandel6"),
    ("tensor", "mandel9"),
    ("mandel9", "tensor"),
    ("mandel9", "mandel6"),
    ("mandel6", "tensor"),
    ("mandel6", "mandel9"),
    ("mandel6", "voigt"),
    ("voigt", "mandel6"),
    ("voigt", "umat"),
    ("voigt", "vumat"),
    ("umat", "voigt"),
    ("vumat", "voigt"),
    ("voigt", alias_abaquas_material_anisotropic),
    (alias_abaquas_material_anisotropic, "voigt"),
]


fig = plt.figure(figsize=(15, 15))
g = netgraph.Graph(edges, node_layout=pos, **plot_options)
g.node_label_artists[alias_abaquas_material_anisotropic].set_size(24),
plt.gca().set_title(label="Stiffness", fontdict=titlefont_options)
plt.tight_layout()
plt.savefig(
    "stiffness_graph.pdf",
    dpi=200,
    bbox_inches="tight",
    pad_inches=0.05,
)
plt.close()

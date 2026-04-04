import numpy as np
import matplotlib.pyplot as plt

def to_layer_activations(x):
    x = np.asarray(x)
    if x.ndim == 1:
        acts = np.abs(x)
    else:
        acts = np.mean(np.abs(x), axis=0)
    return acts.astype(float)

def normalize(v):
    return np.clip(v, 0, 1)

class NeuralNetHelper():
    def plot_activations(input, hidden1_output, hidden2_output, final_output, figsize=(12, 7)):
        layers = [
            normalize(to_layer_activations(input)),
            normalize(to_layer_activations(hidden1_output)),
            normalize(to_layer_activations(hidden2_output)),
            normalize(to_layer_activations(final_output)),
        ]

        layer_names = ["Input", "Hidden 1", "Hidden 2", "Output"]

        fig, ax = plt.subplots(figsize=figsize)
        ax.set_title("Neural Network Activation Diagram", fontsize=16, pad=30)
        ax.axis("off")

        x_positions = np.linspace(0, 1, len(layers))
        node_radius = 0.03

        node_positions = []

        for li, (layer, x) in enumerate(zip(layers, x_positions)):
            n_nodes = len(layer)
            if n_nodes == 1:
                y_positions = np.array([0.5])
            else:
                y_positions = np.linspace(0.1, 0.9, n_nodes)

            current_positions = []
            for yi, act in zip(y_positions, layer):
                alpha = 0.15 + 0.85 * float(act)
                circle = plt.Circle(
                    (x, yi),
                    node_radius,
                    facecolor="black",
                    edgecolor="black",
                    alpha=alpha,
                    linewidth=1.2,
                    zorder=3,
                )
                ax.add_patch(circle)
                
                ax.text(
                    x,
                    yi,
                    f"{act:.2f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white",
                    zorder=4
                )
                current_positions.append((x, yi, float(act)))

            node_positions.append(current_positions)

            ax.text(
                x,
                0.98,
                layer_names[li],
                ha="center",
                va="bottom",
                fontsize=12,
                fontweight="bold",
            )

        for layer_idx in range(len(node_positions) - 1):
            left_layer = node_positions[layer_idx]
            right_layer = node_positions[layer_idx + 1]

            for x1, y1, a1 in left_layer:
                for x2, y2, a2 in right_layer:
                    conn_alpha = 0.05 + 0.75 * float((a1 + a2) / 2.0)
                    ax.plot(
                        [x1 + node_radius, x2 - node_radius],
                        [y1, y2],
                        color="black",
                        alpha=conn_alpha,
                        linewidth=1.0,
                        zorder=1,
                    )

        plt.tight_layout()
        plt.show()
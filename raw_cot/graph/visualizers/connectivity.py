# visualizers/connectivity.py
import os
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import random
from collections import deque
from base_visualizer import BaseGraphVisualizer

class ConnectivityVisualizer(BaseGraphVisualizer):
    """Visualizes Connectivity Check, starting with an adjacency matrix."""

    def __init__(self, output_dir="outputs", graph_type=None, num_nodes=None, directed=None, 
                 layout=None, num_components=None):
        super().__init__(algorithm_name="Connectivity", output_dir=output_dir)
        # Connectivity can be checked on any graph structure
        self.graph_type = graph_type or random.choice(["random_sparse", "random_dense", "disconnected"])
        self.num_nodes = num_nodes  # Will use in _generate_graph_and_matrix if provided
        self.is_directed = directed  # Will use in _generate_graph_and_matrix if provided
        self.num_components = num_components  # Will use for disconnected graphs if provided
        self._generate_graph_and_matrix()
        self.set_layout(layout or random.choice(["spring", "kamada_kawai", "circular"]))

    def _generate_graph_and_matrix(self):
        """Generates a graph and its adjacency matrix."""
        n = self.num_nodes if self.num_nodes is not None else random.randint(6, 10)
        directed = self.is_directed if self.is_directed is not None else random.choice([True, False])

        if self.graph_type == "disconnected":
            # Ensure graph is disconnected
            num_components = self.num_components if self.num_components is not None else random.randint(2, 4)
            nodes_per_comp = n // num_components
            nodes_list = list(range(n))
            random.shuffle(nodes_list)
            self.G = nx.DiGraph() if directed else nx.Graph()
            self.G.add_nodes_from(range(n))
            start_idx = 0
            for i in range(num_components):
                comp_nodes = nodes_list[start_idx : start_idx + nodes_per_comp]
                if not comp_nodes: continue
                # Create a small connected component within these nodes
                subgraph = nx.gnp_random_graph(len(comp_nodes), p=0.6, seed=self.seed + i, directed=directed)
                mapping = {j: comp_nodes[j] for j in range(len(comp_nodes))}
                subgraph = nx.relabel_nodes(subgraph, mapping)
                self.G.add_edges_from(subgraph.edges())
                start_idx += nodes_per_comp
                if i == num_components -1 and start_idx < n: # Add remaining nodes to last component
                     rem_nodes = nodes_list[start_idx:]
                     for node in rem_nodes:
                         if comp_nodes: # Connect to a node in the component
                             connect_to = random.choice(comp_nodes)
                             self.G.add_edge(node, connect_to)
                             if not directed: self.G.add_edge(connect_to, node)


        elif self.graph_type == "random_dense":
             p = random.uniform(0.5, 0.8)
             self.G = nx.gnp_random_graph(n, p=p, seed=self.seed, directed=directed)
        else: # random_sparse
             p = random.uniform(0.1, 0.25)
             self.G = nx.gnp_random_graph(n, p=p, seed=self.seed, directed=directed)

        # Ensure nodes are 0..N-1 integers for matrix indexing
        if not all(isinstance(node, int) for node in self.G.nodes()) or \
           set(self.G.nodes()) != set(range(self.G.number_of_nodes())):
             self.G = nx.convert_node_labels_to_integers(self.G, first_label=0)

        self.n_nodes = self.G.number_of_nodes()
        self.adj_matrix = nx.to_numpy_array(self.G, nodelist=sorted(self.G.nodes()))

        print(f"Generated {self.graph_type} graph ({'directed' if directed else 'undirected'}) with {self.n_nodes} nodes for Connectivity.")


    def plot_adjacency_matrix(self):
        """Plots the adjacency matrix."""
        if self.adj_matrix is None:
            print("Warning: Adjacency matrix not available.")
            return None

        fig, ax = plt.subplots(figsize=(8, 8))
        cmap = 'viridis' if self.theme == 'dark' else 'Greys'
        cax = ax.matshow(self.adj_matrix, cmap=cmap)
        fig.colorbar(cax)

        ax.set_title("Adjacency Matrix", color='white' if self.theme == 'dark' else 'black', pad=20)
        ax.set_xticks(np.arange(self.n_nodes))
        ax.set_yticks(np.arange(self.n_nodes))
        ax.set_xticklabels(np.arange(self.n_nodes))
        ax.set_yticklabels(np.arange(self.n_nodes))
        ax.tick_params(axis='both', which='major', labelsize=10,
                       labelcolor='white' if self.theme=='dark' else 'black',
                       color='white' if self.theme=='dark' else 'black')

        # Add grid lines
        ax.set_xticks(np.arange(self.n_nodes + 1) - 0.5, minor=True)
        ax.set_yticks(np.arange(self.n_nodes + 1) - 0.5, minor=True)
        ax.grid(which='minor', color='grey' if self.theme=='dark' else 'lightgrey', linestyle='-', linewidth=0.5)
        ax.tick_params(which='minor', size=0)


        fname = os.path.join(self.problem_folder, f"step_{len(self.images)+1:03d}_adj_matrix.png")
        plt.savefig(fname, bbox_inches='tight', facecolor=fig.get_facecolor())
        plt.close(fig)
        self.images.append(fname)
        print(f"Saved adjacency matrix plot to {fname}")
        return fname


    def run_connectivity(self):
        """Check graph connectivity and visualize without traversal steps."""
        G = self._ensure_graph_exists()
        self.algorithm_name = "Connectivity"
        self.setup_problem_folder() # Reset folder

        # Step 1: Plot Adjacency Matrix
        self.plot_adjacency_matrix()

        if G.number_of_nodes() == 0:
            self.draw_graph(title="Connectivity Check: Empty Graph")
            return self.save_problem_data(None, {"is_connected": True, "components": []})

        # Step 2: Plot the graph
        self.draw_graph(title="Connectivity Check: Graph Visualization")

        # Directly check connectivity using NetworkX without traversal visualization
        if G.is_directed():
            is_connected = nx.is_strongly_connected(G) if G.number_of_nodes() > 0 else True
            is_strongly_connected = is_connected
            num_strong_components = nx.number_strongly_connected_components(G)
            num_weak_components = nx.number_weakly_connected_components(G)
            components = [list(c) for c in nx.strongly_connected_components(G)]
            connectivity_type = "Strongly Connected" if is_connected else "Not Strongly Connected"
            print(f"Directed graph: {connectivity_type}, {num_strong_components} strong components")
        else:
            is_connected = nx.is_connected(G) if G.number_of_nodes() > 0 else True
            components = [list(c) for c in nx.connected_components(G)]
            num_weak_components = len(components)
            connectivity_type = "Connected" if is_connected else "Disconnected"
            print(f"Undirected graph: {connectivity_type}, {num_weak_components} components")

        # Step 3: Show final connectivity result
        final_title = f"Connectivity Result: {connectivity_type}"
        final_desc = f"Graph has {len(components)} {'strongly ' if G.is_directed() else ''}connected component(s)."
        self.draw_graph(
            title=final_title,
            step_description=final_desc
        )

        result_data = {
            "is_connected": is_connected,
            "components": components,
            "is_strongly_connected": is_strongly_connected if G.is_directed() else is_connected,
            "num_components": len(components),
            "num_strongly_connected_components": num_strong_components if G.is_directed() else len(components),
            "num_weakly_connected_components": num_weak_components if G.is_directed() else num_weak_components
        }

        return self.save_problem_data(None, result_data)
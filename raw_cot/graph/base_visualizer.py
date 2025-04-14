# base_visualizer.py
import matplotlib.pyplot as plt
import numpy as np
import random
import networkx as nx
import os
import json
from datetime import datetime
import math

# --- Helper function for colors (same as before) ---
def get_random_color_palette():
    """Generates a random palette for graph elements."""
    hue = random.random()
    saturation = random.uniform(0.6, 0.9)
    lightness_node = random.uniform(0.6, 0.8)
    lightness_visited = random.uniform(0.4, 0.6)
    lightness_current = random.uniform(0.3, 0.5)
    lightness_edge = random.uniform(0.7, 0.9)
    lightness_path = random.uniform(0.5, 0.7)

    def hsl_to_hex(h, s, l):
        c = (1 - abs(2 * l - 1)) * s; x = c * (1 - abs((h * 6) % 2 - 1)); m = l - c / 2
        r, g, b = 0, 0, 0
        if 0 <= h * 6 < 1: r, g, b = c, x, 0
        elif 1 <= h * 6 < 2: r, g, b = x, c, 0
        elif 2 <= h * 6 < 3: r, g, b = 0, c, x
        elif 3 <= h * 6 < 4: r, g, b = 0, x, c
        elif 4 <= h * 6 < 5: r, g, b = x, 0, c
        elif 5 <= h * 6 < 6: r, g, b = c, 0, x
        r, g, b = int((r + m) * 255), int((g + m) * 255), int((b + m) * 255)
        return f'#{r:02x}{g:02x}{b:02x}'

    visited_hue = (hue + 0.3) % 1.0; current_hue = (hue + 0.6) % 1.0; path_hue = (hue + 0.7) % 1.0
    return {
        "node": hsl_to_hex(hue, saturation, lightness_node),
        "visited": hsl_to_hex(visited_hue, saturation, lightness_visited),
        "current": hsl_to_hex(current_hue, saturation, lightness_current),
        "edge": hsl_to_hex(hue, 0.1, lightness_edge),
        "path": hsl_to_hex(path_hue, 0.9, lightness_path),
        "cycle": "#FF6347", # Tomato red for cycles
        "mst_edge": hsl_to_hex(path_hue, 0.9, lightness_path),
        "recursion_stack": "#ADD8E6" # Light blue for recursion stack nodes
    }

class BaseGraphVisualizer:
    """Base class for visualizing graph algorithms."""

    def __init__(self, algorithm_name="UnknownAlgorithm", output_dir="outputs"):
        self.algorithm_name = algorithm_name # Set initial default algo name
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.seed = random.randint(1, 10000)
        random.seed(self.seed)
        np.random.seed(self.seed)

        self.G = None
        self.n_nodes = 0
        self.graph_type = "custom" # Default, subclasses should override

        self.images = []
        self.problem_folder = None # Will be set by setup_problem_folder
        self.layout_type = "spring"
        self.pos = None

        # Visualization Options
        self.theme = "light"
        self.size_by_degree = False
        self.color_edges_by_weight = False
        self._node_shape_marker = "o"
        self.palette = get_random_color_palette()
        self.edge_style = "solid"

        # REMOVED: self.setup_problem_folder() from here

    def setup_problem_folder(self):
        """Create a unique folder for the current problem instance.
           Call this at the START of specific run_... methods."""
        # Ensure graph_type is set by subclass if not default
        graph_type_name = self.graph_type if self.graph_type != "custom" else "graph" # Use generic if not set
        safe_algo_name = "".join(c if c.isalnum() else "_" for c in self.algorithm_name)
        self.problem_folder = os.path.join(
            self.output_dir,
            # Use the specific graph_type set by the subclass here
            f"{graph_type_name}_{safe_algo_name}_{self.timestamp}"
        )
        os.makedirs(self.problem_folder, exist_ok=True)
        self.images = [] # Reset images for the new problem run
        print(f"Set up output folder: {self.problem_folder}") # Add confirmation

    def set_theme(self, theme):
        valid_themes = ["light", "dark"]
        self.theme = theme if theme in valid_themes else "light"
        return self

    def set_size_by_degree(self, enabled=True):
        self.size_by_degree = enabled
        return self

    def set_color_edges_by_weight(self, enabled=True):
        self.color_edges_by_weight = enabled
        return self

    def set_edge_style(self, style):
        valid_styles = ["solid", "dashed", "dotted", "dashdot"]
        self.edge_style = style if style in valid_styles else "solid"
        return self

    def set_layout(self, layout_type):
        valid_layouts = ["spring", "circular", "shell", "spectral", "kamada_kawai", "planar", "random"]
        self.layout_type = layout_type if layout_type in valid_layouts else "spring"
        self.pos = None # Reset position cache
        return self

    def _ensure_graph_exists(self):
        if self.G is None:
            raise ValueError("Graph object 'self.G' has not been generated or assigned.")
        return self.G

    def _get_layout_pos(self):
        """Get or compute node positions."""
        if self.pos is None:
            G = self._ensure_graph_exists()
            layout_func_map = {
                "spring": nx.spring_layout, "circular": nx.circular_layout, "shell": nx.shell_layout,
                "spectral": nx.spectral_layout, "kamada_kawai": nx.kamada_kawai_layout,
                "planar": nx.planar_layout, "random": nx.random_layout,
            }
            layout_func = layout_func_map.get(self.layout_type, nx.spring_layout)
            # Only add seed parameter for layouts that support it
            kwargs = {'seed': self.seed} if self.layout_type in ["spring", "random"] else {}

            # Handle layout failures/warnings
            try:
                if self.layout_type == "planar":
                     if not nx.is_planar(G): raise nx.NetworkXException("Graph is not planar")
                self.pos = layout_func(G, **kwargs)
            except Exception as e:
                print(f"Warning: Layout '{self.layout_type}' failed or not suitable ({e}). Falling back to 'spring' layout.")
                self.layout_type = "spring" # Reset type to avoid repeated errors
                self.pos = nx.spring_layout(G, seed=self.seed)
        return self.pos

    def draw_graph(self, title="", visited=None, current_node=None,
                  highlight_edges=None, highlight_nodes=None, recursion_stack=None,
                  queue_info=None, dist_info=None, step_description=None, extra_text=None):
        """Draw the graph with current state visualization."""
        G = self._ensure_graph_exists()
        pos = self._get_layout_pos()

        plt.style.use('dark_background' if self.theme == "dark" else 'default')
        fig, ax = plt.subplots(figsize=(14, 10))
        bg_color = "#1A1B25" if self.theme == "dark" else "#FFFFFF"
        text_color = "#E0E0E0" if self.theme == "dark" else "#000000"
        fig.patch.set_facecolor(bg_color)
        ax.set_facecolor(bg_color)

        # --- Node Drawing with increased size ---
        base_node_size = 1500  # Increased from 1000 to 1500
        if self.size_by_degree:
            node_size = [base_node_size + 400 * G.degree(n) for n in G.nodes()]  # Increased multiplier from 300 to 400
        else:
            node_list = list(G.nodes())
            node_size = [base_node_size] * len(node_list) if node_list else []

        # Node coloring logic (same as before)
        node_colors = [self.palette["node"]] * G.number_of_nodes()
        if recursion_stack:
            for node in recursion_stack:
                if 0 <= node < len(node_colors): node_colors[node] = self.palette["recursion_stack"]
        if visited:
            for node in visited:
                 # Check bounds and ensure it doesn't override recursion_stack color unless specifically intended
                 if 0 <= node < len(node_colors) and node_colors[node] == self.palette["node"]:
                      node_colors[node] = self.palette["visited"]
        if highlight_nodes:
             for node in highlight_nodes:
                  if 0 <= node < len(node_colors): node_colors[node] = self.palette["cycle"]
        if current_node is not None:
            if 0 <= current_node < len(node_colors): node_colors[current_node] = self.palette["current"]

        nx.draw_networkx_nodes(G, pos, ax=ax, node_size=node_size,
                             node_color=node_colors, node_shape=self._node_shape_marker,
                             linewidths=1.5, edgecolors=text_color)
        nx.draw_networkx_labels(G, pos, ax=ax, font_size=14, font_color=text_color, font_weight='bold')  # Increased from 11 to 14

        # --- Edge Drawing ---
        edge_color = self.palette["edge"]
        base_edge_width = 1.5
        edge_widths = [base_edge_width] * G.number_of_edges()
        weights_present = all('weight' in G[u][v] for u, v in G.edges())

        # Prepare arguments, conditionally add arrow args
        edge_draw_kwargs = {
            "G": G, "pos": pos, "ax": ax, "alpha": 0.6,
            "width": edge_widths, "edge_color": edge_color,
            "style": self.edge_style, "arrows": G.is_directed()
        }
        highlight_draw_kwargs = {
             "G": G, "pos": pos, "ax": ax, "width": 4.0,
             "style": 'solid', "alpha": 0.9, "arrows": G.is_directed()
        }

        # Add arrow style/size only if directed
        if G.is_directed():
            edge_draw_kwargs['arrowstyle'] = '-|>'
            edge_draw_kwargs['arrowsize'] = 15
            highlight_draw_kwargs['arrowstyle'] = '-|>'
            highlight_draw_kwargs['arrowsize'] = 20


        if weights_present:
            weights = [G[u][v].get('weight', 1) for u, v in G.edges()]
            max_w = max(weights) if weights else 1; min_w = min(weights) if weights else 1
            if max_w == min_w: max_w +=1
            edge_widths = [base_edge_width + 3 * ((w - min_w) / (max_w - min_w)) for w in weights]
            edge_draw_kwargs['width'] = edge_widths # Update width

            if self.color_edges_by_weight:
                norm = plt.Normalize(min_w, max_w); cmap = plt.cm.viridis
                edge_colors_cmap = [cmap(norm(w)) for w in weights]
                edge_draw_kwargs['edge_color'] = edge_colors_cmap # Update edge color

            # Draw base edges
            nx.draw_networkx_edges(**edge_draw_kwargs) # Use updated kwargs

            # Draw edge labels with larger font size
            edge_labels = {(u, v): d['weight'] for u, v, d in G.edges(data=True) if 'weight' in d}
            try:
                nx.draw_networkx_edge_labels(G, pos, ax=ax, edge_labels=edge_labels,
                                          font_color=text_color, font_size=12,  # Increased from 9 to 12
                                          bbox=dict(facecolor=bg_color, alpha=0.7, edgecolor='none', pad=0.3))  # Increased padding
            except Exception as e:
                print(f"Warning: Could not draw edge labels: {e}")
        else:
            # Draw unweighted base edges
            nx.draw_networkx_edges(**edge_draw_kwargs)


        # Highlight specific edges
        if highlight_edges:
            edge_list = [(u, v) for u, v, *_ in highlight_edges]
            highlight_color_key = "mst_edge" if "MST" in self.algorithm_name else "path"
            highlight_draw_kwargs['edgelist'] = edge_list
            highlight_draw_kwargs['edge_color'] = self.palette.get(highlight_color_key, self.palette["path"])

            nx.draw_networkx_edges(**highlight_draw_kwargs) # Use updated kwargs


        # --- Info Text (same as before) ---
        info_texts = []
        if title: info_texts.append(title)
        if step_description: info_texts.append(f"Action: {step_description}")
        if queue_info: info_texts.append(f"Queue/Stack/PQ: {queue_info}")
        if dist_info: info_texts.append(f"Distances: {dist_info}")
        if extra_text: info_texts.append(extra_text)

        for i, text in enumerate(info_texts):
             ax.text(0.5, -0.02 - i*0.04, text, ha='center', va='top',
                    transform=ax.transAxes, fontsize=11, color=text_color, wrap=True,
                    bbox=dict(boxstyle='round,pad=0.3', fc=bg_color, ec='none', alpha=0.8))

        ax.axis('off')
        plt.tight_layout(rect=[0, 0.05, 1, 0.95])

        # --- Save Figure (same as before) ---
        step_num = len(self.images) + 1
        fname = os.path.join(self.problem_folder, f"step_{step_num:03d}.png")
        plt.savefig(fname, bbox_inches='tight', facecolor=fig.get_facecolor(), dpi=120)
        plt.close(fig)
        self.images.append(fname)

    def draw_final_solution(self, title, result_data, highlight_nodes=None, highlight_edges=None):
        """
        Draw a clean final solution image showing only the result.
        
        Args:
            title (str): Title of the final solution
            result_data (dict): Algorithm result data to display
            highlight_nodes (list): Nodes to highlight in the final solution
            highlight_edges (list): Edges to highlight in the final solution
        """
        G = self._ensure_graph_exists()
        pos = self._get_layout_pos()
        
        plt.style.use('dark_background' if self.theme == "dark" else 'default')
        fig, ax = plt.subplots(figsize=(16, 12))  # Larger figure for final solution
        bg_color = "#1A1B25" if self.theme == "dark" else "#FFFFFF"
        text_color = "#E0E0E0" if self.theme == "dark" else "#000000"
        fig.patch.set_facecolor(bg_color)
        ax.set_facecolor(bg_color)
        
        # Draw nodes with larger size for final solution
        base_node_size = 1800
        if self.size_by_degree:
            node_size = [base_node_size + 400 * G.degree(n) for n in G.nodes()]
        else:
            node_list = list(G.nodes())
            node_size = [base_node_size] * len(node_list) if node_list else []
        
        # Node coloring for final solution
        node_colors = [self.palette["node"]] * G.number_of_nodes()
        if highlight_nodes:
            for node in highlight_nodes:
                if 0 <= node < len(node_colors):
                    node_colors[node] = self.palette["visited"]
        
        # Draw nodes
        nx.draw_networkx_nodes(G, pos, ax=ax, node_size=node_size,
                             node_color=node_colors, node_shape=self._node_shape_marker,
                             linewidths=2.0, edgecolors=text_color)
        nx.draw_networkx_labels(G, pos, ax=ax, font_size=14, font_color=text_color, font_weight='bold')
        
        # Draw edges with light color for non-solution edges
        edge_color = self.palette["edge"]
        base_edge_width = 1.0  # Thinner for non-solution edges
        
        # Draw all edges with low alpha
        edge_draw_kwargs = {
            "G": G, "pos": pos, "ax": ax, "alpha": 0.3,
            "width": base_edge_width, "edge_color": edge_color,
            "style": 'dotted', "arrows": G.is_directed()
        }
        
        if G.is_directed():
            edge_draw_kwargs['arrowstyle'] = '-|>'
            edge_draw_kwargs['arrowsize'] = 15
        
        # Draw base edges with low alpha
        nx.draw_networkx_edges(**edge_draw_kwargs)
        
        # Highlight solution edges with stronger color and thicker lines
        if highlight_edges:
            edge_list = [(u, v) for u, v, *_ in highlight_edges]
            highlight_color_key = "mst_edge" if "MST" in self.algorithm_name else "path"
            highlight_draw_kwargs = {
                "G": G, "pos": pos, "ax": ax, "width": 5.0,  # Thicker for solution edges
                "style": 'solid', "alpha": 1.0, "arrows": G.is_directed(),
                "edgelist": edge_list,
                "edge_color": self.palette.get(highlight_color_key, self.palette["path"])
            }
            
            if G.is_directed():
                highlight_draw_kwargs['arrowstyle'] = '-|>'
                highlight_draw_kwargs['arrowsize'] = 20
            
            nx.draw_networkx_edges(**highlight_draw_kwargs)
            
            # Draw edge labels for solution edges only
            if all('weight' in G[u][v] for u, v in edge_list):
                edge_labels = {(u, v): G[u][v]['weight'] for u, v in edge_list}
                nx.draw_networkx_edge_labels(G, pos, ax=ax, edge_labels=edge_labels,
                                           font_color=text_color, font_size=14,
                                           bbox=dict(facecolor=bg_color, alpha=0.9, edgecolor='none', pad=0.3))
        
        # Add solution summary as text
        summary_text = title + "\n"
        if isinstance(result_data, dict):
            for key, value in result_data.items():
                if key not in ['components', 'has_cycle', 'predecessors']:  # Skip complex data
                    summary_text += f"{key}: {value}\n"
        
        ax.text(0.5, -0.05, summary_text, ha='center', va='top',
               transform=ax.transAxes, fontsize=14, color=text_color, wrap=True,
               bbox=dict(boxstyle='round,pad=0.5', fc=bg_color, ec='none', alpha=0.9))
        
        ax.axis('off')
        plt.tight_layout(rect=[0, 0.05, 1, 0.95])
        
        # Save as special final solution image
        fname = os.path.join(self.problem_folder, "final_solution.png")
        plt.savefig(fname, bbox_inches='tight', facecolor=fig.get_facecolor(), dpi=150)
        plt.close(fig)
        self.images.append(fname)
        print(f"Saved final solution image to {fname}")
        return fname

    def save_problem_data(self, start_node, result, additional_info=None):
        """Save problem details and visualization info to JSON."""
        if not self.problem_folder: return None
        G = self._ensure_graph_exists()

        adj_list = {i: list(sorted(G.neighbors(i))) for i in G.nodes()}
        weighted_adj = None
        weights_present = all('weight' in G[u][v] for u, v in G.edges())
        if weights_present:
             weighted_adj = {i: sorted([(j, G[i][j]['weight']) for j in G.neighbors(i)]) for i in G.nodes()}

        problem_desc = f"Apply {self.algorithm_name} to a {self.graph_type} graph."
        if start_node is not None: problem_desc += f" Start node: {start_node}."
        problem_desc += f" Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}."

        # Ensure result is JSON serializable
        def make_serializable(data):
            if isinstance(data, dict):
                return {k: make_serializable(v) for k, v in data.items()}
            elif isinstance(data, (list, tuple)):
                return [make_serializable(item) for item in data]
            elif isinstance(data, set):
                return sorted(list(make_serializable(item) for item in data))
            elif isinstance(data, (np.int64, np.int32)):
                return int(data)
            elif isinstance(data, (np.float64, np.float32)):
                 return float(data)
            elif data == float('inf'):
                 return "Infinity"
            else:
                return data

        serializable_result = make_serializable(result)
        serializable_additional = make_serializable(additional_info)

        problem_data = {
            "problem_description": problem_desc, "algorithm": self.algorithm_name,
            "graph_type": self.graph_type, "num_nodes": G.number_of_nodes(),
            "num_edges": G.number_of_edges(), "is_directed": G.is_directed(),
            "seed_used": self.seed,
            "image_files": [os.path.basename(img) for img in self.images],
            "solution_summary": {
                "start_node": start_node,
                "result": serializable_result,
            },
            "graph_data": { "adjacency_list": adj_list, "weighted_adjacency_list": weighted_adj },
            "visualization_settings": {
                "layout": self.layout_type, "theme": self.theme, "node_shape": "circle",
                "palette": self.palette, "edge_style": self.edge_style,
                "size_by_degree": self.size_by_degree,
                "color_edges_by_weight": self.color_edges_by_weight and weights_present
            }
        }
        if serializable_additional:
            problem_data["solution_summary"].update(serializable_additional)

        json_path = os.path.join(self.problem_folder, "problem_details.json")
        try:
            with open(json_path, "w") as f:
                json.dump(problem_data, f, indent=2)
        except TypeError as e:
             print(f"ERROR: Could not serialize problem data to JSON: {e}")
             # Fallback: Save basic info without complex results
             del problem_data["solution_summary"]
             del problem_data["graph_data"]
             with open(json_path, "w") as f:
                 json.dump(problem_data, f, indent=2)
             print("WARNING: Saved JSON with simplified data due to serialization error.")

        return problem_data
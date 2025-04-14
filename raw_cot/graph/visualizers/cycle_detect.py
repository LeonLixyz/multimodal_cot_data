# visualizers/cycle_detect.py
import networkx as nx
import random
from base_visualizer import BaseGraphVisualizer

class CycleDetectVisualizer(BaseGraphVisualizer):
    """Visualizes Cycle Detection using DFS."""

    def __init__(self, output_dir="outputs"):
        super().__init__(algorithm_name="CycleDetection", output_dir=output_dir)
        # Cycles can occur in various graphs
        self.graph_type = random.choice(["random_maybe_cyclic", "directed_maybe_cyclic"])
        self._generate_graph()
        self.set_layout(random.choice(["spring", "circular", "kamada_kawai"]))

    def _generate_graph(self):
        """Generates a graph that might contain cycles."""
        n = random.randint(6, 10)
        is_directed = (self.graph_type == "directed_maybe_cyclic")
        # Generate a base tree/forest
        self.G = nx.gnp_random_graph(n, p=0.05, seed=self.seed, directed=is_directed)
        # Add extra edges to potentially create cycles
        num_extra_edges = random.randint(n // 3, n // 2)
        added_count = 0
        attempts = 0
        while added_count < num_extra_edges and attempts < n*n:
             u, v = random.sample(range(n), 2)
             if not self.G.has_edge(u, v):
                 # For undirected, don't add reverse if directed edge exists
                 if not is_directed or not self.G.has_edge(v, u):
                     self.G.add_edge(u, v)
                     added_count += 1
             attempts += 1

        self.n_nodes = self.G.number_of_nodes()
        print(f"Generated {self.graph_type} graph with {self.n_nodes} nodes, {self.G.number_of_edges()} edges for Cycle Detection.")


    def run_cycle_detection(self):
        """Run Cycle Detection using DFS, visualizing key steps."""
        G = self._ensure_graph_exists()
        self.algorithm_name = "CycleDetection"
        self.setup_problem_folder()

        self.draw_graph(title="Cycle Detection Initial State")

        visited = set()
        recursion_stack = set()
        step_counter = 1
        cycle_found_edge = None

        def dfs_util(node, parent=None):
            nonlocal step_counter, cycle_found_edge
            if cycle_found_edge: return True # Stop early

            visited.add(node)
            recursion_stack.add(node)

            self.draw_graph(
                title=f"Step {step_counter}: Explore {node}",
                step_description=f"Visit {node}. Stack: {recursion_stack}",
                visited=list(visited), current_node=node,
                recursion_stack=list(recursion_stack) # Highlight stack
            )
            step_counter += 1

            for neighbor in sorted(list(G.neighbors(node))):
                if not G.is_directed() and neighbor == parent: continue # Skip parent in undirected

                if neighbor not in visited:
                    if dfs_util(neighbor, node): return True # Cycle found deeper
                elif neighbor in recursion_stack:
                     cycle_found_edge = (node, neighbor)
                     step_counter += 1
                     # Explicitly visualize the found cycle edge
                     # Reconstruct approximate path for highlighting nodes
                     approx_cycle_nodes = list(recursion_stack)
                     try:
                         # Find path between neighbor and node within the stack (can be complex)
                         # Simple highlight: current stack + neighbor
                         highlight_nodes = list(recursion_stack)
                         if neighbor not in highlight_nodes: highlight_nodes.append(neighbor)
                     except:
                         highlight_nodes = list(recursion_stack)

                     self.draw_graph(
                          title=f"Step {step_counter}: Cycle Detected!",
                          step_description=f"Edge ({node}→{neighbor}) points to node in stack!",
                          visited=list(visited),
                          highlight_nodes=highlight_nodes, # Highlight involved nodes
                          current_node=node,
                          highlight_edges=[cycle_found_edge], # Highlight back edge
                          recursion_stack=list(recursion_stack)
                      )
                     return True

            recursion_stack.remove(node) # Backtrack: remove from stack
            return False

        found = False
        for node in sorted(list(G.nodes())):
            if node not in visited:
                if dfs_util(node):
                    found = True
                    break

        if not found:
             step_counter += 1
             self.draw_graph(title=f"Step {step_counter}: No Cycles Found", visited=list(visited))

        return self.save_problem_data(None, list(cycle_found_edge) if cycle_found_edge else [], {"has_cycle": found})
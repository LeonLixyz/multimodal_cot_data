# visualizers/topo_sort.py
import networkx as nx
import random
from collections import deque
from base_visualizer import BaseGraphVisualizer

class TopoSortVisualizer(BaseGraphVisualizer):
    """Visualizes Topological Sort (Kahn's algorithm)."""

    def __init__(self, output_dir="outputs"):
        super().__init__(algorithm_name="TopologicalSort", output_dir=output_dir)
        self.graph_type = "dag"
        self._generate_graph()
        # DAGs often look good with specific layouts
        self.set_layout(random.choice(["spectral", "spring", "shell"]))

    def _generate_graph(self):
        """Generates a Directed Acyclic Graph (DAG)."""
        n = random.randint(6, 10)
        self.G = nx.DiGraph()
        self.G.add_nodes_from(range(n))
        edge_prob = random.uniform(0.15, 0.35)
        # Add edges only from lower index to higher index to guarantee DAG
        for i in range(n):
            for j in range(i + 1, n):
                if random.random() < edge_prob:
                    self.G.add_edge(i, j)
        self.n_nodes = self.G.number_of_nodes()
        print(f"Generated {self.graph_type} graph with {self.n_nodes} nodes for Topological Sort.")


    def run_topological_sort(self):
        """Run Topological Sort (Kahn's), visualizing key steps."""
        G = self._ensure_graph_exists()
        if not nx.is_directed_acyclic_graph(G):
             print("ERROR: Provided graph is not a DAG. Cannot perform topological sort.")
             self.draw_graph(title="Error: Graph has cycles!", step_description="Topological sort requires a DAG.")
             return self.save_problem_data(None, [], {"error": "Graph contains cycles"})

        self.algorithm_name = "TopologicalSort"
        self.setup_problem_folder()

        in_degree = {node: G.in_degree(node) for node in G.nodes()}
        queue = deque(sorted([node for node, degree in in_degree.items() if degree == 0]))
        result = []
        step = 1

        in_degree_str = ", ".join([f"{n}:{d}" for n,d in in_degree.items()])
        self.draw_graph(
            title=f"Step {step}: Initialization",
            step_description=f"In-degrees: {in_degree_str}. Queue (in=0): {list(queue)}.",
            queue_info=f"{list(queue)}"
        )

        while queue:
            current = queue.popleft()
            result.append(current)
            step += 1
            neighbors_updated = []

            for neighbor in sorted(list(G.successors(current))):
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
                    neighbors_updated.append(f"{neighbor}(in=0!)")
                else:
                     neighbors_updated.append(f"{neighbor}(in={in_degree[neighbor]})")

            queue = deque(sorted(list(queue))) # Keep queue sorted for consistent viz

            desc = f"Dequeue {current}. Add to result. Update neighbors: {', '.join(neighbors_updated)}."
            if any("!" in s for s in neighbors_updated): desc += f" Added new 0-in nodes to queue."

            self.draw_graph(
                title=f"Step {step}: Process Node {current}", step_description=desc,
                visited=result, current_node=current, queue_info=f"{list(queue)}"
            )

        final_title = ""
        final_result = result
        add_info = {"has_cycle": False}
        if len(result) != G.number_of_nodes():
            final_title = "Cycle Detected! Topological Sort Incomplete."
            remaining = set(G.nodes()) - set(result)
            desc = f"Processed: {result}. Remaining (cycle?): {remaining}"
            self.draw_graph(title=final_title, step_description=desc, visited=result)
            add_info["has_cycle"] = True
            final_result = result # Return partial result
        else:
            final_title = f"Topological Sort Complete: {' → '.join(map(str, result))}"
            self.draw_graph(title=final_title, visited=result)
            
        # Generate final clean solution visualization
        final_title = "Topological Sort Order" if not add_info["has_cycle"] else "Topological Sort (Incomplete - Cycle Detected)"
        final_result = {"Order": " → ".join(map(str, final_result)), "Has Cycle": add_info["has_cycle"]}
        self.draw_final_solution(final_title, final_result, highlight_nodes=result)

        return self.save_problem_data(None, final_result, add_info)
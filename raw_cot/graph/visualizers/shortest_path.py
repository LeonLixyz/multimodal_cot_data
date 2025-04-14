# visualizers/shortest_path.py
import networkx as nx
import random
import heapq
from base_visualizer import BaseGraphVisualizer

class ShortestPathVisualizer(BaseGraphVisualizer):
    """Visualizes Dijkstra's algorithm."""

    def __init__(self, output_dir="outputs"):
        super().__init__(algorithm_name="Dijkstra", output_dir=output_dir)
        self.graph_type = random.choice(["random_weighted", "complete_weighted"])
        self._generate_graph()
        self.set_layout(random.choice(["spring", "kamada_kawai", "spectral"]))
        self.set_color_edges_by_weight(True) # Good default for shortest path

    def _generate_graph(self):
        """Generates a weighted graph suitable for Dijkstra."""
        n = random.randint(6, 10)
        if self.graph_type == "complete_weighted":
            self.G = nx.complete_graph(n)
            p = 1.0 # Edge probability
        else: # random_weighted
             p = random.uniform(0.3, 0.6)
             self.G = nx.gnp_random_graph(n, p=p, seed=self.seed, directed=False)
             # Ensure connectivity
             if not nx.is_connected(self.G):
                 components = list(nx.connected_components(self.G))
                 while len(components) > 1:
                      c1, c2 = random.sample(components, 2)
                      u, v = random.choice(list(c1)), random.choice(list(c2))
                      self.G.add_edge(u, v)
                      components = list(nx.connected_components(self.G))

        # Add positive integer weights
        min_w, max_w = 1, random.randint(10, 20)
        for u, v in self.G.edges():
            self.G[u][v]['weight'] = random.randint(min_w, max_w)

        self.n_nodes = self.G.number_of_nodes()
        print(f"Generated {self.graph_type} graph with {self.n_nodes} nodes, edge prob ~{p:.2f}, weights [{min_w}-{max_w}] for Dijkstra.")


    def run_dijkstra(self, start_node=None):
        """Run Dijkstra's algorithm, visualizing key steps."""
        G = self._ensure_graph_exists()
        self.algorithm_name = "Dijkstra"
        self.setup_problem_folder()

        if start_node is None or start_node not in G:
             start_node = random.choice(list(G.nodes()))

        self.draw_graph(title=f"Dijkstra Initial State (Start Node: {start_node})")

        distances = {node: float('inf') for node in G.nodes()}
        predecessors = {node: None for node in G.nodes()}
        distances[start_node] = 0
        pq = [(0, start_node)]
        visited = set()
        step = 1

        dist_str = ", ".join([f"{n}: {d if d != float('inf') else '∞'}" for n, d in distances.items()])
        self.draw_graph(
            title=f"Step {step}: Initialization",
            step_description=f"Distances initialized. Start node {start_node} dist=0. PQ: {pq}",
            visited=list(visited), current_node=start_node,
            queue_info=f"{sorted(pq)}", dist_info=dist_str
        )

        while pq:
            dist, current = heapq.heappop(pq)
            if current in visited: continue
            visited.add(current)
            step += 1
            updates_made = []

            for neighbor in G.neighbors(current):
                if neighbor not in visited:
                    weight = G[current][neighbor]['weight']
                    new_dist = distances[current] + weight
                    if new_dist < distances[neighbor]:
                        old_dist_str = f"{distances[neighbor] if distances[neighbor] != float('inf') else '∞'}"
                        distances[neighbor] = new_dist
                        predecessors[neighbor] = current
                        heapq.heappush(pq, (new_dist, neighbor))
                        updates_made.append(f"{neighbor}({old_dist_str}→{new_dist})")

            dist_str = ", ".join([f"{n}: {d if d != float('inf') else '∞'}" for n, d in distances.items()])
            desc = f"Finalize node {current} (Dist: {dist}). " + (f"Updated neighbors: {', '.join(updates_made)}." if updates_made else "No updates.")

            path_edges = []
            curr = current
            while predecessors[curr] is not None:
                 pred = predecessors[curr]; path_edges.append((pred, curr)); curr = pred

            pq_sorted_str = " ".join([f"({d},{n})" for d, n in sorted(pq)]) # Compact PQ display
            self.draw_graph(
                title=f"Step {step}: Finalize Node {current}", step_description=desc,
                visited=list(visited), current_node=current, highlight_edges=path_edges,
                queue_info=pq_sorted_str, dist_info=dist_str
            )

        final_dist_str = ", ".join([f"{n}: {d if d != float('inf') else '∞'}" for n, d in distances.items()])
        self.draw_graph(title="Dijkstra Algorithm Complete", visited=list(visited), dist_info=final_dist_str)
        
        # Generate final clean solution visualization
        final_path_edges = []
        for node in G.nodes():
            if predecessors[node] is not None:
                final_path_edges.append((predecessors[node], node, distances[node]))
        
        final_title = f"Dijkstra's Shortest Paths from Node {start_node}"
        final_result = {"Source": start_node, "Total Paths": len(final_path_edges)}
        self.draw_final_solution(final_title, final_result, highlight_nodes=[start_node], highlight_edges=final_path_edges)
        
        shortest_paths = {node: distances[node] for node in G.nodes()}
        return self.save_problem_data(start_node, distances, {"predecessors": predecessors}) # Save distances dict
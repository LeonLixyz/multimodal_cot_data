# visualizers/mst.py
import networkx as nx
import random
import heapq
from base_visualizer import BaseGraphVisualizer

class MSTVisualizer(BaseGraphVisualizer):
    """Visualizes Prim's and Kruskal's MST algorithms."""

    def __init__(self, output_dir="outputs"):
        # MST needs weighted, preferably connected graph
        super().__init__(algorithm_name="MST", output_dir=output_dir)
        self.graph_type = random.choice(["random_connected_weighted", "complete_weighted"])
        self._generate_graph()
        self.set_layout(random.choice(["spring", "kamada_kawai", "spectral"]))
        self.set_color_edges_by_weight(True)

    def _generate_graph(self):
        """Generates a weighted graph suitable for MST."""
        n = random.randint(8, 15)
        if self.graph_type == "complete_weighted":
            self.G = nx.complete_graph(n)
            p = 1.0
        else: # random_connected_weighted
             # Start sparse, add edges to ensure connectivity
             p = 0.15
             self.G = nx.gnp_random_graph(n, p=p, seed=self.seed, directed=False)
             if not nx.is_connected(self.G):
                 components = list(nx.connected_components(self.G))
                 while len(components) > 1:
                      c1, c2 = random.sample(components, 2)
                      u, v = random.choice(list(c1)), random.choice(list(c2))
                      self.G.add_edge(u, v)
                      # Add a few more random edges after ensuring connectivity
                      for _ in range(n // 2):
                         u, v = random.sample(range(n), 2)
                         if not self.G.has_edge(u,v): self.G.add_edge(u,v)

                      components = list(nx.connected_components(self.G))


        min_w, max_w = 1, random.randint(10, 25)
        for u, v in self.G.edges():
            self.G[u][v]['weight'] = random.randint(min_w, max_w)

        self.n_nodes = self.G.number_of_nodes()
        print(f"Generated {self.graph_type} graph with {self.n_nodes} nodes, weights [{min_w}-{max_w}] for MST.")


    def run_prim(self, start_node=None):
        """Run Prim's algorithm, visualizing key steps."""
        G = self._ensure_graph_exists()
        self.algorithm_name = "Prim_MST"
        self.setup_problem_folder()

        if start_node is None or start_node not in G:
            start_node = random.choice(list(G.nodes()))

        self.draw_graph(title=f"Prim's MST Initial State (Start Node: {start_node})")

        mst_edges_data = [] # Store (u, v, weight) tuples
        mst_nodes = {start_node}
        edges_pq = []
        total_weight = 0
        step = 1

        for neighbor in G.neighbors(start_node):
            weight = G[start_node][neighbor]['weight']
            heapq.heappush(edges_pq, (weight, start_node, neighbor))

        pq_sorted_str = " ".join([f"({w},{u}→{v})" for w, u, v in sorted(edges_pq)])
        self.draw_graph(
             title=f"Step {step}: Initialization",
             step_description=f"Start at {start_node}. Add edges to neighbors to PQ.",
             visited=list(mst_nodes), current_node=start_node, queue_info=pq_sorted_str
        )

        while edges_pq and len(mst_nodes) < G.number_of_nodes():
            weight, u, v = heapq.heappop(edges_pq)
            if v in mst_nodes: continue

            mst_nodes.add(v)
            mst_edges_data.append((u, v, weight))
            total_weight += weight
            step += 1
            new_edges_added_desc = []

            for neighbor in G.neighbors(v):
                if neighbor not in mst_nodes:
                    neighbor_weight = G[v][neighbor]['weight']
                    heapq.heappush(edges_pq, (neighbor_weight, v, neighbor))
                    new_edges_added_desc.append(f"({neighbor_weight},{v}→{neighbor})")

            pq_sorted_str = " ".join([f"({w},{uu}→{vv})" for w, uu, vv in sorted(edges_pq)])
            desc = f"Add edge ({u}→{v}, w:{weight}). Total: {total_weight}. Node {v} added."
            if new_edges_added_desc: desc += f" Added to PQ: {', '.join(new_edges_added_desc)}."

            self.draw_graph(
                title=f"Step {step}: Add Edge ({u}, {v})", step_description=desc,
                visited=list(mst_nodes), current_node=v, highlight_edges=mst_edges_data,
                queue_info=pq_sorted_str
            )

        final_title = f"Prim's MST Complete. Total Weight: {total_weight}"
        if len(mst_nodes) != G.number_of_nodes(): final_title += " (Graph disconnected?)"
        self.draw_graph(title=final_title, visited=list(mst_nodes), highlight_edges=mst_edges_data)
        
        # Generate final clean solution visualization
        final_title = f"Prim's Minimum Spanning Tree (Total Weight: {total_weight})"
        final_result = {"Start Node": start_node, "Total Weight": total_weight, "Edges": len(mst_edges_data)}
        self.draw_final_solution(final_title, final_result, highlight_nodes=list(mst_nodes), highlight_edges=mst_edges_data)
        
        return self.save_problem_data(start_node, mst_edges_data, {"total_weight": total_weight})

    def run_kruskal(self):
        """Run Kruskal's algorithm, visualizing key steps."""
        G = self._ensure_graph_exists()
        self.algorithm_name = "Kruskal_MST"
        self.setup_problem_folder()

        all_edges = sorted([(G[u][v]['weight'], u, v) for u, v in G.edges()])
        self.draw_graph(
            title="Kruskal's MST Initial State",
            step_description=f"Edges sorted by weight (showing first few): {all_edges[:min(5, len(all_edges))]}..."
            )

        parent = {node: node for node in G.nodes()}
        def find(node):
            if parent[node] == node: return node
            parent[node] = find(parent[node]); return parent[node]
        def union(node1, node2):
            root1, root2 = find(node1), find(node2)
            if root1 != root2: parent[root1] = root2; return True
            return False

        mst_edges_data = []
        total_weight = 0
        step = 1

        for i, (weight, u, v) in enumerate(all_edges):
            step += 1
            root_u, root_v = find(u), find(v)
            accepted = False
            desc = f"Consider edge ({u},{v}, w:{weight}). Find({u})={root_u}, Find({v})={root_v}."

            if root_u != root_v:
                if union(u, v):
                    accepted = True
                    mst_edges_data.append((u, v, weight))
                    total_weight += weight
                    desc += f" Accepted. MST weight: {total_weight}."
            else:
                desc += " Rejected (cycle)."

            self.draw_graph(
                title=f"Step {step}: Consider Edge ({u}, {v})", step_description=desc,
                highlight_edges=mst_edges_data, # Show current MST
                # Optionally: highlight the edge being considered differently
                extra_text=f"Edge considered: ({u}, {v}, w:{weight})" if not accepted else ""
            )

            if len(mst_edges_data) == G.number_of_nodes() - 1: break

        final_title = f"Kruskal's MST Complete. Total Weight: {total_weight}"
        if len(mst_edges_data) != G.number_of_nodes() - 1: final_title += f" (Graph disconnected? Edges: {len(mst_edges_data)})"
        self.draw_graph(title=final_title, highlight_edges=mst_edges_data)
        
        # Generate final clean solution visualization
        final_title = f"Kruskal's Minimum Spanning Tree (Total Weight: {total_weight})"
        final_result = {"Total Weight": total_weight, "Edges": len(mst_edges_data)}
        self.draw_final_solution(final_title, final_result, highlight_edges=mst_edges_data)
        
        return self.save_problem_data(None, mst_edges_data, {"total_weight": total_weight})
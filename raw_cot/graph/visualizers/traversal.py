# visualizers/traversal.py
import networkx as nx
import random
from collections import deque
from base_visualizer import BaseGraphVisualizer # Import from sibling directory/package path

class TraversalVisualizer(BaseGraphVisualizer):
    """Visualizes BFS and DFS graph traversals."""

    def __init__(self, output_dir="outputs", graph_type=None, num_nodes=None, layout=None, start_node=None):
        super().__init__(algorithm_name="Traversal", output_dir=output_dir)
        # Traversal can be performed on various graph structures
        self.graph_type = graph_type or random.choice(["random_tree", "grid", "random_sparse"])
        self.num_nodes = num_nodes  # Will use in _generate_graph if provided
        self.custom_start_node = start_node  # Will use in traversal algorithms if provided
        self._generate_graph()
        self.set_layout(layout or random.choice(["spring", "kamada_kawai", "spectral"]))

    def _generate_graph(self):
        """Generates a graph suitable for traversal."""
        n = self.num_nodes if self.num_nodes is not None else random.randint(6, 10)
        if self.graph_type == "random_tree":
            # Generate a random tree using Max Spanning Tree on a complete graph
             G_comp = nx.complete_graph(n)
             for u, v in G_comp.edges():
                 G_comp[u][v]['weight'] = random.randint(1, 15)
             self.G = nx.maximum_spanning_tree(G_comp, weight='weight')
             self.n_nodes = self.G.number_of_nodes()

        elif self.graph_type == "grid":
            dim = int(n**0.5)
            self.G = nx.grid_2d_graph(dim, dim)
            mapping = {node: i for i, node in enumerate(sorted(self.G.nodes()))}
            self.G = nx.relabel_nodes(self.G, mapping)
            self.n_nodes = self.G.number_of_nodes()

        else: # random_sparse
             self.G = nx.gnp_random_graph(n, p=0.2, seed=self.seed, directed=False)
             # Ensure connectivity for traversal demo
             if not nx.is_connected(self.G):
                 components = list(nx.connected_components(self.G))
                 while len(components) > 1:
                      c1, c2 = random.sample(components, 2)
                      u, v = random.choice(list(c1)), random.choice(list(c2))
                      self.G.add_edge(u, v)
                      components = list(nx.connected_components(self.G))
             self.n_nodes = self.G.number_of_nodes()

        print(f"Generated {self.graph_type} graph with {self.n_nodes} nodes for Traversal.")

    def run_bfs(self, start_node=None):
        """Run BFS, visualizing key steps."""
        G = self._ensure_graph_exists()
        self.algorithm_name = "BFS" # Be specific for saving
        self.setup_problem_folder() # Reset folder for specific algo run

        if start_node is None or start_node not in G:
            start_node = random.choice(list(G.nodes()))

        self.draw_graph(title=f"BFS Initial State (Start Node: {start_node})")

        visited = {start_node}
        queue = deque([start_node])
        step = 1

        self.draw_graph(
            title=f"Step {step}: Start BFS",
            step_description=f"Start at {start_node}. Add to queue and visited.",
            visited=list(visited), current_node=start_node, queue_info=f"{list(queue)}"
        )

        while queue:
            current = queue.popleft()
            step += 1
            neighbors_added = []
            for neighbor in sorted(list(G.neighbors(current))):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
                    neighbors_added.append(neighbor)

            desc = f"Dequeue {current}. " + (f"Visit/enqueue neighbors: {neighbors_added}." if neighbors_added else "No new neighbors.")
            self.draw_graph(
                title=f"Step {step}: Process Node {current}", step_description=desc,
                visited=list(visited), current_node=current, queue_info=f"{list(queue)}"
            )

        self.draw_graph(title="BFS Traversal Complete", visited=list(visited))
        
        # Generate final clean solution visualization
        final_title = f"BFS Traversal from Node {start_node}"
        final_result = {"Start Node": start_node, "Visited Nodes": len(visited)}
        self.draw_final_solution(final_title, final_result, highlight_nodes=list(visited))
        
        return self.save_problem_data(start_node, list(visited)) # Pass result directly

    def run_dfs_recursive(self, start_node=None):
        """Run Recursive DFS, visualizing key steps."""
        G = self._ensure_graph_exists()
        self.algorithm_name = "DFS_Recursive" # Set specific algorithm name
        self.setup_problem_folder() # Reset folder for specific algo run

        if start_node is None or start_node not in G:
            start_node = random.choice(list(G.nodes()))

        self.draw_graph(title=f"Recursive DFS Initial State (Start Node: {start_node})")

        visited = set()
        traversal_path = []
        recursion_stack_visual = []
        step_counter = 1

        def dfs_util(node, parent=None):
            nonlocal step_counter
            visited.add(node)
            traversal_path.append(node)
            recursion_stack_visual.append(node)

            self.draw_graph(
                title=f"Step {step_counter}: Explore {node}",
                step_description=f"Enter {node}" + (f" from {parent}" if parent is not None else " (start)"),
                visited=list(visited), current_node=node,
                recursion_stack=list(recursion_stack_visual), # Pass stack for coloring
                queue_info=f"{recursion_stack_visual}" # Show stack content
            )
            step_counter += 1

            for neighbor in sorted(list(G.neighbors(node))):
                is_parent_edge = (not G.is_directed() and neighbor == parent)
                if not is_parent_edge and neighbor not in visited:
                        dfs_util(neighbor, node)

            recursion_stack_visual.pop()
            self.draw_graph(
                title=f"Step {step_counter}: Backtrack from {node}",
                step_description=f"Finished {node}. Backtracking" + (f" to {parent}" if parent is not None else " (end)."),
                visited=list(visited), current_node=node, # Show node we are leaving
                recursion_stack=list(recursion_stack_visual),
                queue_info=f"{recursion_stack_visual}"
            )
            step_counter += 1

        dfs_util(start_node)
        self.draw_graph(title="Recursive DFS Complete", visited=list(visited))
        
        # Generate final clean solution visualization
        final_title = f"DFS Traversal from Node {start_node}"
        final_result = {"Start Node": start_node, "Visited Nodes": len(visited), "Traversal Order": " → ".join(map(str, traversal_path))}
        self.draw_final_solution(final_title, final_result, highlight_nodes=list(visited))
        
        return self.save_problem_data(start_node, traversal_path)

if __name__ == "__main__":
    # Use argparse for better command-line argument handling
    parser = argparse.ArgumentParser(description="Run graph algorithm visualizations")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Parser for 'all' command - run all examples
    all_parser = subparsers.add_parser("all", help="Run all algorithm examples")
    all_parser.add_argument("--output-dir", help="Output directory for visualizations")
    
    # Parser for 'connectivity' command - run predefined connectivity examples
    conn_parser = subparsers.add_parser("connectivity", help="Run predefined connectivity examples")
    conn_parser.add_argument("--output-dir", help="Output directory for visualizations")
    
    # Parser for 'traversal' command - run predefined traversal examples
    trav_parser = subparsers.add_parser("traversal", help="Run predefined traversal examples")
    trav_parser.add_argument("--output-dir", help="Output directory for visualizations")
    
    # Add other parsers as needed...
    
    # Parse arguments
    args = parser.parse_args()
    
    # Handle commands
    if len(sys.argv) == 1 or args.command is None:
        # No command specified, default to 'all'
        main_output_dir = "graph_algorithm_examples"
        run_all_examples(output_base_dir=main_output_dir)
    elif args.command == "help":
        print_help()
    elif args.command == "all":
        output_dir = args.output_dir or "graph_algorithm_examples"
        run_all_examples(output_base_dir=output_dir)
    elif args.command == "connectivity":
        output_dir = args.output_dir or "connectivity_examples"
        run_connectivity_examples(output_base_dir=output_dir)
    elif args.command == "traversal":
        output_dir = args.output_dir or "traversal_examples"
        run_traversal_examples(output_base_dir=output_dir)
    # Other commands...
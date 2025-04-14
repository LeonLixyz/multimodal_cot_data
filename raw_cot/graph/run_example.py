#!/usr/bin/env python3
"""
Graph Algorithm Visualizer - Run examples and visualizations for various graph algorithms.
"""
import os
import sys
import argparse
import itertools
from typing import Dict, List, Any, Optional, Union, Callable

# Add the parent directory to sys.path if needed
parent_dir = os.path.dirname(os.path.abspath(__file__))

# Import visualizer classes
from visualizers.traversal import TraversalVisualizer
from visualizers.shortest_path import ShortestPathVisualizer
from visualizers.mst import MSTVisualizer
from visualizers.topo_sort import TopoSortVisualizer
from visualizers.cycle_detect import CycleDetectVisualizer
from visualizers.connectivity import ConnectivityVisualizer


# --------- Algorithm Runners ---------

def run_traversal(output_dir: str, graph_type: str = "random_sparse", 
                 num_nodes: int = 10, algorithm: str = "bfs", 
                 layout: str = "spring", **kwargs) -> None:
    """Run traversal algorithm visualization."""
    try:
        trav_viz = TraversalVisualizer(
            output_dir=output_dir,
            graph_type=graph_type,
            num_nodes=num_nodes,
            layout=layout
        )
        
        if algorithm == "bfs":
            trav_viz.run_bfs()
        elif algorithm == "dfs":
            trav_viz.run_dfs_recursive()
        else:
            print(f"Unknown traversal algorithm: {algorithm}")
    except Exception as e:
        print(f"Error running traversal algorithm {algorithm} on {graph_type}: {e}")


def run_shortest_path(output_dir: str, algorithm: str = "dijkstra", **kwargs) -> None:
    """Run shortest path algorithm visualization."""
    try:
        sp_viz = ShortestPathVisualizer(output_dir=output_dir)
        
        if algorithm == "dijkstra":
            sp_viz.run_dijkstra()
        else:
            print(f"Unknown shortest path algorithm: {algorithm}")
    except Exception as e:
        print(f"Error running shortest path algorithm {algorithm}: {e}")


def run_mst(output_dir: str, algorithm: str = "prim", **kwargs) -> None:
    """Run minimum spanning tree algorithm visualization."""
    try:
        mst_viz = MSTVisualizer(output_dir=output_dir)
        
        if algorithm == "prim":
            mst_viz.run_prim()
        elif algorithm == "kruskal":
            mst_viz.run_kruskal()
        else:
            print(f"Unknown MST algorithm: {algorithm}")
    except Exception as e:
        print(f"Error running MST algorithm {algorithm}: {e}")


def run_topo_sort(output_dir: str, **kwargs) -> None:
    """Run topological sort algorithm visualization."""
    try:
        topo_viz = TopoSortVisualizer(output_dir=output_dir)
        topo_viz.run_topological_sort()
    except Exception as e:
        print(f"Error running topological sort: {e}")


def run_cycle_detection(output_dir: str, **kwargs) -> None:
    """Run cycle detection algorithm visualization."""
    try:
        cycle_viz = CycleDetectVisualizer(output_dir=output_dir)
        cycle_viz.run_cycle_detection()
    except Exception as e:
        print(f"Error running cycle detection: {e}")


def run_connectivity(output_dir: str, graph_type: str = "random_sparse", 
                    num_nodes: int = 8, directed: bool = False,
                    layout: str = "spring", num_components: Optional[int] = None, 
                    **kwargs) -> None:
    """Run connectivity algorithm visualization."""
    try:
        params = {
            "output_dir": output_dir,
            "graph_type": graph_type,
            "num_nodes": num_nodes,
            "directed": directed,
            "layout": layout
        }
        
        # Only add num_components if graph_type is disconnected
        if graph_type == "disconnected" and num_components is not None:
            params["num_components"] = num_components
        
        conn_viz = ConnectivityVisualizer(**params)
        conn_viz.run_connectivity()
    except Exception as e:
        print(f"Error running connectivity algorithm on {graph_type}: {e}")


# --------- Main Command Functions ---------

def run_all(output_dir: str = "graph_algorithm_examples") -> None:
    """Run all algorithm examples with default parameters."""
    print("\n=== Running All Graph Algorithm Examples ===")
    os.makedirs(output_dir, exist_ok=True)
    
    # Traversal algorithms
    graph_types = ["random_tree", "grid", "random_sparse"]
    algorithms = ["bfs", "dfs"]
    
    for graph_type in graph_types:
        for algorithm in algorithms:
            algo_name = "BFS" if algorithm == "bfs" else "DFS"
            print(f"\n-- Running {algo_name} on {graph_type} graph --")
            
            run_traversal(
                output_dir=os.path.join(output_dir, f"traversal_{algorithm}_{graph_type}"),
                graph_type=graph_type,
                algorithm=algorithm
            )
    
    # Shortest path
    print("\n-- Running Shortest Path (Dijkstra) --")
    run_shortest_path(output_dir=os.path.join(output_dir, "shortest_path_dijkstra"))
    
    # MST algorithms
    for algorithm in ["prim", "kruskal"]:
        print(f"\n-- Running MST ({algorithm.capitalize()}) --")
        run_mst(output_dir=os.path.join(output_dir, f"mst_{algorithm}"), algorithm=algorithm)
    
    # Topological Sort
    print("\n-- Running Topological Sort --")
    run_topo_sort(output_dir=os.path.join(output_dir, "topological_sort"))
    
    # Cycle Detection
    print("\n-- Running Cycle Detection --")
    run_cycle_detection(output_dir=os.path.join(output_dir, "cycle_detection"))
    
    # Connectivity examples
    conn_configs = [
        {"graph_type": "random_sparse", "directed": False},
        {"graph_type": "random_dense", "directed": True},
        {"graph_type": "disconnected", "directed": False, "num_components": 3}
    ]
    
    for i, config in enumerate(conn_configs):
        print(f"\n-- Running Connectivity Example {i+1}: {config['graph_type']} --")
        run_connectivity(
            output_dir=os.path.join(output_dir, f"connectivity_{i+1}_{config['graph_type']}"),
            **config
        )
    
    print(f"\n=== All Examples Completed ===")
    print(f"Results saved to: {output_dir}")


def run_specific(args: argparse.Namespace) -> None:
    """Run a specific algorithm with custom parameters."""
    algorithm = args.algorithm
    output_dir = args.output_dir or f"{algorithm}_output"
    os.makedirs(output_dir, exist_ok=True)
    
    # Common parameters to extract from args
    params = {
        "output_dir": output_dir,
        "graph_type": getattr(args, "graph_type", "random_sparse"),
        "num_nodes": getattr(args, "num_nodes", 10),
        "layout": getattr(args, "layout", "spring"),
        "directed": getattr(args, "directed", False),
    }
    
    # Add algorithm-specific parameter if provided
    if hasattr(args, "sub_algorithm") and args.sub_algorithm:
        params["algorithm"] = args.sub_algorithm
    
    # Add num_components if provided for connectivity
    if hasattr(args, "num_components") and args.num_components:
        params["num_components"] = args.num_components
    
    print(f"\n=== Running {algorithm.capitalize()} Algorithm ===")
    print("Parameters:")
    for k, v in params.items():
        if k != "output_dir":
            print(f"- {k}: {v}")
    
    # Map the algorithm name to the appropriate function
    algorithm_map = {
        "traversal": run_traversal,
        "shortest_path": run_shortest_path,
        "mst": run_mst,
        "topo_sort": run_topo_sort,
        "cycle_detection": run_cycle_detection,
        "connectivity": run_connectivity,
    }
    
    if algorithm in algorithm_map:
        algorithm_map[algorithm](**params)
        print(f"\n=== Visualization Complete ===")
        print(f"Results saved to: {output_dir}")
    else:
        print(f"Unknown algorithm: {algorithm}")
        print("Available algorithms: " + ", ".join(algorithm_map.keys()))


def run_combinations(args: argparse.Namespace) -> None:
    """Run parameter combinations for a specific algorithm."""
    algorithm = args.algorithm
    output_dir = args.output_dir or f"{algorithm}_combinations"
    
    # Parameter options for different algorithms
    parameter_options = {
        "traversal": {
            "graph_types": ["random_tree", "grid", "random_sparse"],
            "num_nodes": [5, 9, 16],
            "layouts": ["spring", "circular", "kamada_kawai"],
            "algorithms": ["bfs", "dfs"]
        },
        "connectivity": {
            "graph_types": ["random_sparse", "random_dense", "disconnected"],
            "num_nodes": [5, 8, 12],
            "directed": [True, False],
            "layouts": ["spring", "circular", "kamada_kawai"],
            "num_components": [2, 3, 4]
        },
        "mst": {
            "num_nodes": [5, 10, 15],
            "layouts": ["spring", "circular"],
            "algorithms": ["prim", "kruskal"]
        }
    }
    
    # Check if algorithm is supported for combinations
    if algorithm not in parameter_options:
        print(f"Combinations not supported for algorithm: {algorithm}")
        print("Supported algorithms: " + ", ".join(parameter_options.keys()))
        return
    
    params = parameter_options[algorithm]
    
    # Calculate total combinations
    combinations = []
    
    if algorithm == "traversal":
        total = len(params["graph_types"]) * len(params["num_nodes"]) * len(params["layouts"]) * len(params["algorithms"])
        
        print(f"\n=== Running Traversal Algorithm Combinations ===")
        print(f"Total combinations: {total}")
        
        # Ask for confirmation before running all combinations
        response = input(f"This will generate {total} visualizations. Continue? (y/n): ")
        if response.lower() not in ["y", "yes"]:
            print("Operation cancelled.")
            return
        
        # Generate combinations
        counter = 0
        for graph_type in params["graph_types"]:
            for num_nodes in params["num_nodes"]:
                for layout in params["layouts"]:
                    for algorithm in params["algorithms"]:
                        counter += 1
                        
                        combo_dir = os.path.join(
                            output_dir, 
                            f"{counter:03d}_{graph_type}_{num_nodes}nodes_{layout}_{algorithm}"
                        )
                        os.makedirs(combo_dir, exist_ok=True)
                        
                        print(f"\n-- Running combination {counter}/{total}: "
                              f"graph_type={graph_type}, num_nodes={num_nodes}, "
                              f"layout={layout}, algorithm={algorithm} --")
                        
                        run_traversal(
                            output_dir=combo_dir,
                            graph_type=graph_type,
                            num_nodes=num_nodes,
                            layout=layout,
                            algorithm=algorithm
                        )
    
    elif algorithm == "connectivity":
        # Calculate total combinations
        disconnected_count = len(params["num_nodes"]) * len(params["directed"]) * len(params["layouts"]) * len(params["num_components"])
        other_count = len(params["num_nodes"]) * len(params["directed"]) * len(params["layouts"]) * (len(params["graph_types"]) - 1)
        total = disconnected_count + other_count
        
        print(f"\n=== Running Connectivity Algorithm Combinations ===")
        print(f"Total combinations: {total}")
        
        # Ask for confirmation before running all combinations
        response = input(f"This will generate {total} visualizations. Continue? (y/n): ")
        if response.lower() not in ["y", "yes"]:
            print("Operation cancelled.")
            return
        
        # Generate combinations
        counter = 0
        
        # Disconnected graphs with components parameter
        for num_nodes in params["num_nodes"]:
            for directed in params["directed"]:
                for layout in params["layouts"]:
                    for num_components in params["num_components"]:
                        counter += 1
                        
                        combo_dir = os.path.join(
                            output_dir, 
                            f"{counter:03d}_disconnected_{num_nodes}nodes_"
                            f"{'directed' if directed else 'undirected'}_"
                            f"{layout}_{num_components}comps"
                        )
                        os.makedirs(combo_dir, exist_ok=True)
                        
                        print(f"\n-- Running combination {counter}/{total}: "
                              f"graph_type=disconnected, num_nodes={num_nodes}, "
                              f"directed={directed}, layout={layout}, num_components={num_components} --")
                        
                        run_connectivity(
                            output_dir=combo_dir,
                            graph_type="disconnected",
                            num_nodes=num_nodes,
                            directed=directed,
                            layout=layout,
                            num_components=num_components
                        )
        
        # Other graph types without components parameter
        for graph_type in [g for g in params["graph_types"] if g != "disconnected"]:
            for num_nodes in params["num_nodes"]:
                for directed in params["directed"]:
                    for layout in params["layouts"]:
                        counter += 1
                        
                        combo_dir = os.path.join(
                            output_dir, 
                            f"{counter:03d}_{graph_type}_{num_nodes}nodes_"
                            f"{'directed' if directed else 'undirected'}_"
                            f"{layout}"
                        )
                        os.makedirs(combo_dir, exist_ok=True)
                        
                        print(f"\n-- Running combination {counter}/{total}: "
                              f"graph_type={graph_type}, num_nodes={num_nodes}, "
                              f"directed={directed}, layout={layout} --")
                        
                        run_connectivity(
                            output_dir=combo_dir,
                            graph_type=graph_type,
                            num_nodes=num_nodes,
                            directed=directed,
                            layout=layout
                        )
    
    elif algorithm == "mst":
        total = len(params["num_nodes"]) * len(params["layouts"]) * len(params["algorithms"])
        
        print(f"\n=== Running MST Algorithm Combinations ===")
        print(f"Total combinations: {total}")
        
        # Ask for confirmation before running all combinations
        response = input(f"This will generate {total} visualizations. Continue? (y/n): ")
        if response.lower() not in ["y", "yes"]:
            print("Operation cancelled.")
            return
        
        # Generate combinations
        counter = 0
        for num_nodes in params["num_nodes"]:
            for layout in params["layouts"]:
                for algorithm in params["algorithms"]:
                    counter += 1
                    
                    combo_dir = os.path.join(
                        output_dir, 
                        f"{counter:03d}_{num_nodes}nodes_{layout}_{algorithm}"
                    )
                    os.makedirs(combo_dir, exist_ok=True)
                    
                    print(f"\n-- Running combination {counter}/{total}: "
                          f"num_nodes={num_nodes}, layout={layout}, algorithm={algorithm} --")
                    
                    run_mst(
                        output_dir=combo_dir,
                        num_nodes=num_nodes,
                        layout=layout,
                        algorithm=algorithm
                    )
    
    print(f"\n=== All {total} combinations completed ===")
    print(f"Results saved to: {output_dir}")


# --------- Command Line Interface ---------

def setup_parser() -> argparse.ArgumentParser:
    """Set up the command line argument parser."""
    parser = argparse.ArgumentParser(
        description="Graph Algorithm Visualizer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all algorithm examples
  python run_example.py all
  
  # Run a specific traversal algorithm
  python run_example.py specific --algorithm traversal --graph-type random_tree --sub-algorithm bfs
  
  # Run connectivity on a directed disconnected graph
  python run_example.py specific --algorithm connectivity --graph-type disconnected --directed --num-components 3
  
  # Run all combinations for a specific algorithm
  python run_example.py combinations --algorithm traversal
"""
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # 'all' command
    all_parser = subparsers.add_parser("all", help="Run all algorithm examples")
    all_parser.add_argument("--output-dir", help="Output directory for visualizations")
    
    # 'specific' command
    specific_parser = subparsers.add_parser("specific", help="Run a specific algorithm with custom parameters")
    specific_parser.add_argument("--algorithm", required=True, 
                                choices=["traversal", "shortest_path", "mst", "topo_sort", "cycle_detection", "connectivity"],
                                help="Algorithm to run")
    specific_parser.add_argument("--output-dir", help="Output directory for visualizations")
    specific_parser.add_argument("--graph-type", choices=["random_tree", "grid", "random_sparse", "random_dense", "disconnected"],
                                help="Type of graph to use")
    specific_parser.add_argument("--num-nodes", type=int, help="Number of nodes in the graph")
    specific_parser.add_argument("--layout", choices=["spring", "circular", "kamada_kawai", "spectral"],
                                help="Graph layout algorithm")
    specific_parser.add_argument("--directed", action="store_true", help="Make the graph directed")
    specific_parser.add_argument("--sub-algorithm", help="Sub-algorithm to use (e.g., 'bfs' for traversal, 'prim' for MST)")
    specific_parser.add_argument("--num-components", type=int, help="Number of components for disconnected graphs")
    
    # 'combinations' command
    combinations_parser = subparsers.add_parser("combinations", help="Run parameter combinations for an algorithm")
    combinations_parser.add_argument("--algorithm", required=True, 
                                   choices=["traversal", "connectivity", "mst"],
                                   help="Algorithm to run combinations for")
    combinations_parser.add_argument("--output-dir", help="Output directory for visualizations")
    
    return parser


def main() -> None:
    """Main entry point for the application."""
    parser = setup_parser()
    args = parser.parse_args()
    
    if not args.command or args.command == "help":
        parser.print_help()
    elif args.command == "all":
        run_all(output_dir=args.output_dir if args.output_dir else "graph_algorithm_examples")
    elif args.command == "specific":
        run_specific(args)
    elif args.command == "combinations":
        run_combinations(args)
    else:
        print(f"Unknown command: {args.command}")
        parser.print_help()


if __name__ == "__main__":
    main()
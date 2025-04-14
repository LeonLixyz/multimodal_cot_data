# Graph Algorithm 

Currently we have the following algorithms:

- Traversal Algorithms: BFS, DFS
- Shortest Path: Dijkstra's Algorithm
- Minimum Spanning Trees: Prim's Algorithm, Kruskal's Algorithm
- Topological Sorting: Topological Sorting
- Cycle Detection: Cycle Detection
- Connectivity Analysis: Connectivity Analysis

The graph types we have are:

- Random Tree: Tree structures with no cycles
- Grid: Regular grid-structured graphs
- Random Sparse: Graphs with relatively few edges
- Random Dense: Graphs with many edges
- Disconnected: Graphs with multiple connected components

TODO:
- check the correctness of the graphs. Examples are in the folder `graph_algorithm_examples`.
- add two more problem types: maxflow and matching problems.
- see if we can decrease the number of graphs per each trace. Currently we have on average ~15 images per trace. (with the exception of the connectivity problems, which just have 1 trace.)
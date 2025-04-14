import matplotlib.pyplot as plt
import numpy as np
import random
import os
import json
from datetime import datetime
import math
from matplotlib.patches import FancyArrowPatch

class DataStructureVisualizer:
    def __init__(self, output_dir="ds_outputs"):
        """Initialize the data structure visualizer with visualization options."""
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Create timestamp for unique folder name
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Store visualization data
        self.images = []
        self.problem_folder = None
        self.ds_type = "unknown"  # Track data structure type
        
        # Enhanced visualization options
        self.color_scheme = "default"  # default, colorful, monochrome, contrast
        self.node_shape = "circle"     # circle, square, triangle, diamond, hexagon
        self.edge_style = "solid"      # solid, dashed, dotted, dashdot
        self.theme = "light"           # light, dark
        self.show_details = True       # Show additional operation details
        
    # ===== VISUALIZATION SETTING METHODS =====
    
    def set_color_scheme(self, scheme):
        """Set the color scheme for visualization."""
        valid_schemes = ["default", "colorful", "monochrome", "contrast"]
        if scheme not in valid_schemes:
            raise ValueError(f"Invalid color scheme. Choose from: {valid_schemes}")
        self.color_scheme = scheme
        return self
        
    def set_node_shape(self, shape):
        """Set the node shape for visualization."""
        valid_shapes = ["circle", "square", "triangle", "diamond", "hexagon"]
        if shape not in valid_shapes:
            raise ValueError(f"Invalid node shape. Choose from: {valid_shapes}")
        self.node_shape = shape
        return self
        
    def set_edge_style(self, style):
        """Set the edge style for visualization."""
        valid_styles = ["solid", "dashed", "dotted", "dashdot"]
        if style not in valid_styles:
            raise ValueError(f"Invalid edge style. Choose from: {valid_styles}")
        self.edge_style = style
        return self
        
    def set_theme(self, theme):
        """Set the visualization theme."""
        valid_themes = ["light", "dark"]
        if theme not in valid_themes:
            raise ValueError(f"Invalid theme. Choose from: {valid_themes}")
        self.theme = theme
        return self
        
    def set_show_details(self, enabled=True):
        """Set whether to show detailed operation information."""
        self.show_details = enabled
        return self
    
    # ===== UTILITY METHODS =====
    
    def setup_problem_folder(self, operation_name):
        """Create a unique folder for the current problem."""
        self.problem_folder = os.path.join(
            self.output_dir, 
            f"{self.ds_type}_{operation_name}_{self.timestamp}"
        )
        os.makedirs(self.problem_folder, exist_ok=True)
        self.images = []
        return self.problem_folder
    
    def save_image(self, title=""):
        """Save the current figure to the problem folder."""
        if not self.problem_folder:
            raise ValueError("Problem folder must be set up first.")
            
        fname = os.path.join(self.problem_folder, f"step{len(self.images)+1}.png")
        plt.savefig(fname, bbox_inches='tight')
        plt.close()
        self.images.append(fname)
        return fname
        
    def save_problem_data(self, operation, result, additional_info=None):
        """Save problem data to JSON."""
        if not self.problem_folder:
            return None
            
        # Basic problem description
        problem_desc = f"Apply the {operation} operation on the {self.ds_type}."
        
        # Save problem data
        problem_data = {
            "problem": problem_desc,
            "images": [os.path.basename(img) for img in self.images],
            "solution": {
                "operation": operation,
                "result": result,
                "visualization": {
                    "color_scheme": self.color_scheme,
                    "node_shape": self.node_shape,
                    "edge_style": self.edge_style,
                    "theme": self.theme,
                    "show_details": self.show_details
                }
            }
        }
        
        # Add additional info if provided
        if additional_info:
            problem_data["solution"].update(additional_info)
        
        # Save problem data to JSON file
        with open(os.path.join(self.problem_folder, "problem_data.json"), "w") as f:
            json.dump(problem_data, f, indent=2)
        
        return problem_data
    
    # ===== BINARY SEARCH TREE IMPLEMENTATIONS =====
    
    class BSTNode:
        """Binary Search Tree Node class."""
        def __init__(self, key):
            self.key = key
            self.left = None
            self.right = None
            self.height = 1  # For AVL tree
            self.color = "RED"  # For Red-Black tree (RED or BLACK)
            self.parent = None  # For certain operations
    
    def create_bst(self, keys=None, size=None):
        """Create a Binary Search Tree with the given keys or random keys."""
        self.ds_type = "binary_search_tree"
        self.root = None
        
        if keys is None:
            if size is None:
                size = random.randint(5, 15)
            keys = random.sample(range(1, 100), size)
            
        # Insert keys into BST
        for key in keys:
            self.root = self._bst_insert(self.root, key)
            
        return self
        
    def _bst_insert(self, root, key):
        """Insert a key into the BST."""
        # Base case: empty subtree
        if root is None:
            return self.BSTNode(key)
            
        # Recursive insertion based on BST property
        if key < root.key:
            root.left = self._bst_insert(root.left, key)
            if root.left.parent is None:
                root.left.parent = root
        elif key > root.key:
            root.right = self._bst_insert(root.right, key)
            if root.right.parent is None:
                root.right.parent = root
        
        # Return updated root
        return root
    
    def visualize_tree(self, current_node=None, highlighted_nodes=None, 
                      operation_details=None, title=""):
        """Visualize a tree data structure."""
        if highlighted_nodes is None:
            highlighted_nodes = []
            
        # Color scheme definitions
        color_schemes = {
            "default": {
                "node": "lightblue",
                "highlighted": "red",
                "edge": "black",
                "null": "lightgray",
                "text": "black"
            },
            "colorful": {
                "node": "#87CEEB",  # SkyBlue
                "highlighted": "#FF6347",  # Tomato
                "edge": "#4682B4",  # SteelBlue
                "null": "#D3D3D3",  # LightGray
                "text": "#000080"   # Navy
            },
            "monochrome": {
                "node": "#E0E0E0",
                "highlighted": "#404040",
                "edge": "#A0A0A0", 
                "null": "#F5F5F5",
                "text": "#000000"
            },
            "contrast": {
                "node": "#FFF0C9",
                "highlighted": "#D64550",
                "edge": "#1A1B25",
                "null": "#EFEFEF",
                "text": "#000000"
            }
        }
        
        # Set up figure with theme
        plt.figure(figsize=(12, 8))
        
        if self.theme == "dark":
            plt.style.use("dark_background")
            text_color = "white"
            
            # Override text colors for dark theme
            for scheme in color_schemes:
                color_schemes[scheme]["text"] = "white"
                if scheme == "monochrome":
                    color_schemes[scheme]["node"] = "#505050"
                    color_schemes[scheme]["null"] = "#303030"
        else:
            plt.style.use("default")
            text_color = "black"
            
        # Get colors from scheme
        scheme = color_schemes.get(self.color_scheme, color_schemes["default"])
        
        # Calculate positions for tree nodes
        positions = {}
        self._calculate_positions(self.root, positions)
        
        # Set axes off and title
        plt.axis('off')
        plt.title(title, fontsize=16, color=text_color)
        
        # Draw edges first (so they're under the nodes)
        self._draw_edges(self.root, positions, scheme, current_node, highlighted_nodes)
        
        # Draw nodes
        self._draw_nodes(self.root, positions, scheme, current_node, highlighted_nodes)
        
        # Add operation details if provided and enabled
        if operation_details and self.show_details:
            detail_text = "\n".join(operation_details)
            plt.figtext(0.5, 0.02, detail_text, ha='center', fontsize=12, 
                      color=text_color, bbox=dict(facecolor='none', edgecolor=scheme["edge"], 
                                              alpha=0.5, boxstyle='round,pad=1'))
        
        # Save the visualization
        self.save_image()
    
    def _calculate_positions(self, node, positions, x=0, y=0, width=1.0, level=1):
        """Calculate positions for tree nodes using recursive approach."""
        if node is None:
            return
            
        # Store position
        positions[node] = (x, y)
        
        # Calculate offset for next level based on tree width
        level_width = width / 2 * 0.8  # 0.8 is a scaling factor for better spacing
        
        # Calculate positions for children
        self._calculate_positions(node.left, positions, x - level_width, y - 1, level_width, level + 1)
        self._calculate_positions(node.right, positions, x + level_width, y - 1, level_width, level + 1)
    
    def _draw_edges(self, node, positions, color_scheme, current_node=None, highlighted_nodes=None):
        """Draw edges between nodes."""
        if node is None:
            return
            
        # Edge style
        style = {
            "solid": "-",
            "dashed": "--",
            "dotted": ":",
            "dashdot": "-."
        }.get(self.edge_style, "-")
        
        # Draw edges to children
        if node.left:
            start_pos = positions[node]
            end_pos = positions[node.left]
            
            # Determine color based on highlighted status
            edge_color = color_scheme["highlighted"] if (node in highlighted_nodes and node.left in highlighted_nodes) else color_scheme["edge"]
            
            # Draw the edge
            plt.plot([start_pos[0], end_pos[0]], [start_pos[1], end_pos[1]], 
                   style, color=edge_color, linewidth=1.5)
                   
        if node.right:
            start_pos = positions[node]
            end_pos = positions[node.right]
            
            # Determine color based on highlighted status
            edge_color = color_scheme["highlighted"] if (node in highlighted_nodes and node.right in highlighted_nodes) else color_scheme["edge"]
            
            # Draw the edge
            plt.plot([start_pos[0], end_pos[0]], [start_pos[1], end_pos[1]], 
                   style, color=edge_color, linewidth=1.5)
        
        # Recursively draw edges for children
        self._draw_edges(node.left, positions, color_scheme, current_node, highlighted_nodes)
        self._draw_edges(node.right, positions, color_scheme, current_node, highlighted_nodes)
    
    def _draw_nodes(self, node, positions, color_scheme, current_node=None, highlighted_nodes=None):
        """Draw tree nodes."""
        if node is None:
            return
            
        # Node shape parameters
        node_shapes = {
            "circle": "o",
            "square": "s", 
            "triangle": "^", 
            "diamond": "d",
            "hexagon": "h"
        }
        shape = node_shapes.get(self.node_shape, "o")
        
        # Node position
        x, y = positions[node]
        
        # Determine node color based on status
        if node == current_node:
            node_color = color_scheme["highlighted"]
        elif node in highlighted_nodes:
            node_color = color_scheme["highlighted"]
        else:
            node_color = color_scheme["node"]
            
        # Special handling for Red-Black tree colors
        if hasattr(node, 'color') and self.ds_type == "red_black_tree":
            if node.color == "RED":
                node_color = "red" if self.theme == "light" else "#FF6666"
            else:  # BLACK
                node_color = "black" if self.theme == "light" else "#333333"
        
        # Draw the node
        plt.scatter(x, y, s=800, c=node_color, marker=shape, zorder=10)
        
        # Draw the node text
        plt.text(x, y, str(node.key), ha='center', va='center', fontsize=12,
               color=color_scheme["text"], fontweight='bold', zorder=11)
        
        # Draw null children as smaller, lighter nodes
        if node.left is None and (node == current_node or node in highlighted_nodes):
            null_x, null_y = x - 0.5, y - 1
            plt.scatter(null_x, null_y, s=300, c=color_scheme["null"], marker=shape, alpha=0.5, zorder=5)
            plt.text(null_x, null_y, "NIL", ha='center', va='center', fontsize=10, color=color_scheme["text"], alpha=0.7, zorder=6)
            
        if node.right is None and (node == current_node or node in highlighted_nodes):
            null_x, null_y = x + 0.5, y - 1
            plt.scatter(null_x, null_y, s=300, c=color_scheme["null"], marker=shape, alpha=0.5, zorder=5)
            plt.text(null_x, null_y, "NIL", ha='center', va='center', fontsize=10, color=color_scheme["text"], alpha=0.7, zorder=6)
        
        # Recursively draw nodes for children
        self._draw_nodes(node.left, positions, color_scheme, current_node, highlighted_nodes)
        self._draw_nodes(node.right, positions, color_scheme, current_node, highlighted_nodes)
        
    def bst_insert(self, key):
        """Visualize the insertion process in a BST."""
        self.setup_problem_folder("insert_operation")
        
        # Initial tree
        self.visualize_tree(title=f"Initial Binary Search Tree\nInserting key: {key}")
        
        # Trace the insertion path
        path = []
        current = self.root
        
        while current is not None:
            path.append(current)
            
            if key < current.key:
                if current.left is None:
                    break
                current = current.left
            elif key > current.key:
                if current.right is None:
                    break
                current = current.right
            else:
                # Key already exists
                self.visualize_tree(current, path, 
                                  [f"Key {key} already exists in the tree."], 
                                  f"Insert Operation - Key {key} Already Exists")
                return self.save_problem_data("Insert", 
                                            {"key": key, "result": "already_exists"})
        
        # Visualize the search path
        self.visualize_tree(current, path, 
                          [f"Searching for insertion position for key {key}"], 
                          f"Insert Operation - Finding Position for Key {key}")
        
        # Create new node and insert
        new_node = self.BSTNode(key)
        
        if self.root is None:
            self.root = new_node
            self.visualize_tree(new_node, [new_node], 
                              [f"Tree was empty. Key {key} inserted as root."], 
                              f"Insert Operation - Key {key} Inserted as Root")
        else:
            if key < current.key:
                current.left = new_node
            else:
                current.right = new_node
                
            new_node.parent = current
            path.append(new_node)
            
            self.visualize_tree(new_node, path, 
                              [f"Key {key} inserted as {'left' if key < current.key else 'right'} child of {current.key}."], 
                              f"Insert Operation - Key {key} Inserted")
        
        # Final tree
        self.visualize_tree(title=f"Final Binary Search Tree After Inserting {key}")
        
        return self.save_problem_data("Insert", {"key": key, "result": "success"})

    def bst_search(self, key):
        """Visualize the search process in a BST."""
        self.setup_problem_folder("search_operation")
        
        # Initial tree
        self.visualize_tree(title=f"Initial Binary Search Tree\nSearching for key: {key}")
        
        # Trace the search path
        path = []
        current = self.root
        step = 1
        
        while current is not None:
            path.append(current)
            
            self.visualize_tree(current, path, 
                              [f"Step {step}: Examining node with key {current.key}",
                               f"Looking for key {key}",
                               f"{'Found!' if current.key == key else ('Go left' if key < current.key else 'Go right')}"], 
                              f"Search Operation - Step {step}")
            
            step += 1
            
            if key == current.key:
                # Key found
                self.visualize_tree(current, path, 
                                  [f"Key {key} found in the tree."], 
                                  f"Search Operation - Key {key} Found")
                return self.save_problem_data("Search", 
                                            {"key": key, "result": "found"})
            elif key < current.key:
                current = current.left
            else:
                current = current.right
        
        # Key not found
        self.visualize_tree(None, path, 
                          [f"Key {key} not found in the tree."], 
                          f"Search Operation - Key {key} Not Found")
        
        return self.save_problem_data("Search", {"key": key, "result": "not_found"})
        
    def bst_delete(self, key):
        """Visualize the deletion process in a BST."""
        self.setup_problem_folder("delete_operation")
        
        # Initial tree
        self.visualize_tree(title=f"Initial Binary Search Tree\nDeleting key: {key}")
        
        # Find the node to delete
        path = []
        current = self.root
        parent = None
        
        # Search for the node to delete
        while current is not None and current.key != key:
            path.append(current)
            parent = current
            
            if key < current.key:
                current = current.left
            else:
                current = current.right
                
            # Visualize the search path
            self.visualize_tree(current, path, 
                              [f"Searching for key {key} to delete", 
                               f"Current node: {current.key if current else 'None'}", 
                               f"Parent node: {parent.key if parent else 'None'}"], 
                              f"Delete Operation - Searching for Key {key}")
        
        # If key not found
        if current is None:
            self.visualize_tree(None, path, 
                              [f"Key {key} not found in the tree."], 
                              f"Delete Operation - Key {key} Not Found")
            return self.save_problem_data("Delete", 
                                        {"key": key, "result": "not_found"})
        
        # Found the node to delete
        path.append(current)
        self.visualize_tree(current, path, 
                          [f"Found key {key} to delete", 
                           f"Node type: {'Leaf' if (current.left is None and current.right is None) else ('One child' if (current.left is None or current.right is None) else 'Two children')}"], 
                          f"Delete Operation - Found Key {key}")
        
        # Case 1: Node is a leaf (no children)
        if current.left is None and current.right is None:
            if current != self.root:
                if parent.left == current:
                    parent.left = None
                else:
                    parent.right = None
            else:
                self.root = None
                
            self.visualize_tree(parent, [parent] if parent else [], 
                              [f"Node {key} was a leaf node and has been deleted."], 
                              f"Delete Operation - Deleted Leaf Node {key}")
        
        # Case 2: Node has one child
        elif current.left is None:
            # Has right child
            if current != self.root:
                if parent.left == current:
                    parent.left = current.right
                else:
                    parent.right = current.right
                current.right.parent = parent
            else:
                self.root = current.right
                self.root.parent = None
                
            self.visualize_tree(current.right, path, 
                              [f"Node {key} had only a right child.",
                               f"Right child {current.right.key} replaces node {key}."], 
                              f"Delete Operation - Replaced Node {key} with Right Child")
        
        elif current.right is None:
            # Has left child
            if current != self.root:
                if parent.left == current:
                    parent.left = current.left
                else:
                    parent.right = current.left
                current.left.parent = parent
            else:
                self.root = current.left
                self.root.parent = None
                
            self.visualize_tree(current.left, path, 
                              [f"Node {key} had only a left child.",
                               f"Left child {current.left.key} replaces node {key}."], 
                              f"Delete Operation - Replaced Node {key} with Left Child")
        
        # Case 3: Node has two children
        else:
            # Find inorder successor (smallest in right subtree)
            successor_path = path.copy()
            successor_parent = current
            successor = current.right
            successor_path.append(successor)
            
            self.visualize_tree(successor, successor_path, 
                              [f"Node {key} has two children.",
                               f"Finding inorder successor (smallest in right subtree).",
                               f"Starting with right child: {successor.key}"], 
                              f"Delete Operation - Finding Successor for Node {key}")
            
            # Find the leftmost node in the right subtree
            while successor.left is not None:
                successor_parent = successor
                successor = successor.left
                successor_path.append(successor)
                
                self.visualize_tree(successor, successor_path, 
                                  [f"Finding inorder successor for {key}.",
                                   f"Current successor candidate: {successor.key}"], 
                                  f"Delete Operation - Finding Successor")
            
            # Highlight final successor
            self.visualize_tree(successor, successor_path, 
                              [f"Found inorder successor: {successor.key}",
                               f"This value will replace node {key}"], 
                              f"Delete Operation - Found Successor {successor.key}")
            
            # Copy successor value to current node
            current.key = successor.key
            
            # Delete the successor
            if successor_parent != current:
                successor_parent.left = successor.right
                if successor.right:
                    successor.right.parent = successor_parent
            else:
                current.right = successor.right
                if successor.right:
                    successor.right.parent = current
            
            self.visualize_tree(current, path, 
                              [f"Replaced value {key} with successor {successor.key}.",
                               f"Deleted the successor node from its original position."], 
                              f"Delete Operation - Replaced with Successor")
        
        # Final tree
        self.visualize_tree(title=f"Final Binary Search Tree After Deleting {key}")
        
        return self.save_problem_data("Delete", {"key": key, "result": "success"})
    
    def bst_traversal(self, method="inorder"):
        """Visualize a BST traversal."""
        valid_methods = ["inorder", "preorder", "postorder", "levelorder"]
        if method not in valid_methods:
            raise ValueError(f"Invalid traversal method. Choose from: {valid_methods}")
            
        self.setup_problem_folder(f"{method}_traversal")
        
        # Initial tree
        self.visualize_tree(title=f"Initial Binary Search Tree\n{method.capitalize()} Traversal")
        
        # Traversal results
        traversal_order = []
        visited_nodes = []
        
        # Function to perform traversal step by step
        def traverse_step_by_step():
            if method == "inorder":
                self._inorder_traversal(self.root, traversal_order, visited_nodes)
            elif method == "preorder":
                self._preorder_traversal(self.root, traversal_order, visited_nodes)
            elif method == "postorder":
                self._postorder_traversal(self.root, traversal_order, visited_nodes)
            else:  # level order
                self._levelorder_traversal(self.root, traversal_order, visited_nodes)
        
        # Perform traversal
        traverse_step_by_step()
        
        # Final traversal
        self.visualize_tree(None, [], 
                          [f"{method.capitalize()} traversal order: {', '.join(map(str, traversal_order))}"], 
                          f"Final {method.capitalize()} Traversal Result")
        
        return self.save_problem_data(f"{method.capitalize()} Traversal", 
                                    {"order": traversal_order})
    
    def _inorder_traversal(self, node, result, visited_nodes, step=1):
        """Perform inorder traversal with visualization."""
        if node is None:
            return step
            
        # Visit left subtree
        visited_nodes.append(node)
        self.visualize_tree(node, visited_nodes, 
                          [f"Step {step}: At node {node.key}",
                           f"Inorder: First traverse left subtree"], 
                          f"Inorder Traversal - Going Left from {node.key}")
        
        step = self._inorder_traversal(node.left, result, visited_nodes, step + 1)
        
        # Visit current node
        result.append(node.key)
        self.visualize_tree(node, visited_nodes, 
                          [f"Step {step}: Visiting node {node.key}",
                           f"Current traversal: {', '.join(map(str, result))}"], 
                          f"Inorder Traversal - Visiting Node {node.key}")
        
        step += 1
        
        # Visit right subtree
        self.visualize_tree(node, visited_nodes, 
                          [f"Step {step}: At node {node.key}",
                           f"Inorder: Now traverse right subtree"], 
                          f"Inorder Traversal - Going Right from {node.key}")
        
        step = self._inorder_traversal(node.right, result, visited_nodes, step + 1)
        
        return step
    
    def _preorder_traversal(self, node, result, visited_nodes, step=1):
        """Perform preorder traversal with visualization."""
        if node is None:
            return step
            
        # Visit current node first
        visited_nodes.append(node)
        result.append(node.key)
        
        self.visualize_tree(node, visited_nodes, 
                          [f"Step {step}: Visiting node {node.key}",
                           f"Current traversal: {', '.join(map(str, result))}"], 
                          f"Preorder Traversal - Visiting Node {node.key}")
        
        step += 1
        
        # Visit left subtree
        if node.left:
            self.visualize_tree(node, visited_nodes, 
                              [f"Step {step}: At node {node.key}",
                               f"Preorder: Now traverse left subtree"], 
                              f"Preorder Traversal - Going Left from {node.key}")
            
        step = self._preorder_traversal(node.left, result, visited_nodes, step)
        
        # Visit right subtree
        if node.right:
            self.visualize_tree(node, visited_nodes, 
                              [f"Step {step}: At node {node.key}",
                               f"Preorder: Now traverse right subtree"], 
                              f"Preorder Traversal - Going Right from {node.key}")
            
        step = self._preorder_traversal(node.right, result, visited_nodes, step)
        
        return step
    
    def _postorder_traversal(self, node, result, visited_nodes, step=1):
        """Perform postorder traversal with visualization."""
        if node is None:
            return step
            
        # Add node to visited but don't visit yet
        visited_nodes.append(node)
        
        # Visit left subtree
        self.visualize_tree(node, visited_nodes, 
                          [f"Step {step}: At node {node.key}",
                           f"Postorder: First traverse left subtree"], 
                          f"Postorder Traversal - Going Left from {node.key}")
        
        step = self._postorder_traversal(node.left, result, visited_nodes, step + 1)
        
        # Visit right subtree
        self.visualize_tree(node, visited_nodes, 
                          [f"Step {step}: At node {node.key}",
                           f"Postorder: Now traverse right subtree"], 
                          f"Postorder Traversal - Going Right from {node.key}")
        
        step = self._postorder_traversal(node.right, result, visited_nodes, step + 1)
        
        # Visit current node last
        result.append(node.key)
        self.visualize_tree(node, visited_nodes, 
                          [f"Step {step}: Visiting node {node.key}",
                           f"Current traversal: {', '.join(map(str, result))}"], 
                          f"Postorder Traversal - Visiting Node {node.key}")
        
        return step + 1
    
    def _levelorder_traversal(self, root, result, visited_nodes):
        """Perform level order traversal with visualization."""
        if root is None:
            return
            
        # Use a queue for level order traversal
        queue = [root]
        step = 1
        
        while queue:
            # Get the next node from the queue
            node = queue.pop(0)
            visited_nodes.append(node)
            result.append(node.key)
            
            # Visualize current node visit
            self.visualize_tree(node, visited_nodes, 
                              [f"Step {step}: Visiting node {node.key}",
                               f"Current traversal: {', '.join(map(str, result))}"], 
                              f"Level Order Traversal - Visiting Node {node.key}")
            
            step += 1
            
            # Add children to queue
            if node.left:
                queue.append(node.left)
                self.visualize_tree(node, visited_nodes, 
                                  [f"Adding left child {node.left.key} to queue"], 
                                  f"Level Order Traversal - Adding Left Child to Queue")
                
            if node.right:
                queue.append(node.right)
                self.visualize_tree(node, visited_nodes, 
                                  [f"Adding right child {node.right.key} to queue"], 
                                  f"Level Order Traversal - Adding Right Child to Queue")
    
    # ===== AVL TREE IMPLEMENTATIONS =====
    
    def create_avl_tree(self, keys=None, size=None):
        """Create an AVL tree with the given keys or random keys."""
        self.ds_type = "avl_tree"
        self.root = None
        
        if keys is None:
            if size is None:
                size = random.randint(5, 15)
            keys = random.sample(range(1, 100), size)
            
        # Insert keys into AVL tree
        for key in keys:
            self.root = self._avl_insert(self.root, key)
            
        return self
    
    def _get_height(self, node):
        """Get height of an AVL node."""
        if node is None:
            return 0
        return node.height
    
    def _get_balance(self, node):
        """Get balance factor of an AVL node."""
        if node is None:
            return 0
        return self._get_height(node.left) - self._get_height(node.right)
    
    def _avl_rotate_right(self, y, path=None, visualize=False):
        """Perform right rotation on AVL subtree rooted at y."""
        x = y.left
        T2 = x.right
        
        # Visualize before rotation if requested
        if visualize and path:
            path.append(x)
            self.visualize_tree(y, path, 
                              [f"Performing right rotation at node {y.key}",
                               f"Node {x.key} will become the new subtree root"], 
                              f"AVL Tree - Right Rotation at {y.key}")
        
        # Perform rotation
        x.right = y
        y.left = T2
        
        # Update parent pointers
        x.parent = y.parent
        y.parent = x
        if T2:
            T2.parent = y
        
        # Update heights
        y.height = 1 + max(self._get_height(y.left), self._get_height(y.right))
        x.height = 1 + max(self._get_height(x.left), self._get_height(x.right))
        
        # Visualize after rotation if requested
        if visualize and path:
            self.visualize_tree(x, path, 
                              [f"Completed right rotation",
                               f"New subtree root: {x.key}"], 
                              f"AVL Tree - After Right Rotation")
        
        # Return new subtree root
        return x
    
    def _avl_rotate_left(self, x, path=None, visualize=False):
        """Perform left rotation on AVL subtree rooted at x."""
        y = x.right
        T2 = y.left
        
        # Visualize before rotation if requested
        if visualize and path:
            path.append(y)
            self.visualize_tree(x, path, 
                              [f"Performing left rotation at node {x.key}",
                               f"Node {y.key} will become the new subtree root"], 
                              f"AVL Tree - Left Rotation at {x.key}")
        
        # Perform rotation
        y.left = x
        x.right = T2
        
        # Update parent pointers
        y.parent = x.parent
        x.parent = y
        if T2:
            T2.parent = x
        
        # Update heights
        x.height = 1 + max(self._get_height(x.left), self._get_height(x.right))
        y.height = 1 + max(self._get_height(y.left), self._get_height(y.right))
        
        # Visualize after rotation if requested
        if visualize and path:
            self.visualize_tree(y, path, 
                              [f"Completed left rotation",
                               f"New subtree root: {y.key}"], 
                              f"AVL Tree - After Left Rotation")
        
        # Return new subtree root
        return y
    
    def _avl_insert(self, node, key, path=None, visualize=False):
        """Insert a key into the AVL tree."""
        # Standard BST insert
        if node is None:
            new_node = self.BSTNode(key)
            if path is not None:
                path.append(new_node)
            if visualize:
                self.visualize_tree(new_node, path, 
                                  [f"Inserted new node with key {key}"], 
                                  f"AVL Tree - Node {key} Inserted")
            return new_node
            
        if path is not None:
            path.append(node)
            
        if key < node.key:
            node.left = self._avl_insert(node.left, key, path, visualize)
            if node.left.parent is None:
                node.left.parent = node
        elif key > node.key:
            node.right = self._avl_insert(node.right, key, path, visualize)
            if node.right.parent is None:
                node.right.parent = node
        else:
            # Duplicate keys not allowed
            if path is not None and path and path[-1] == node:
                path.pop()
            return node
        
        # Update height
        node.height = 1 + max(self._get_height(node.left), self._get_height(node.right))
        
        # Get balance factor
        balance = self._get_balance(node)
        
        if visualize:
            self.visualize_tree(node, path, 
                              [f"Checking balance at node {node.key}",
                               f"Height of left subtree: {self._get_height(node.left)}",
                               f"Height of right subtree: {self._get_height(node.right)}",
                               f"Balance factor: {balance}"], 
                              f"AVL Tree - Checking Balance at {node.key}")
        
        # Left Left Case
        if balance > 1 and key < node.left.key:
            return self._avl_rotate_right(node, path, visualize)
        
        # Right Right Case
        if balance < -1 and key > node.right.key:
            return self._avl_rotate_left(node, path, visualize)
        
        # Left Right Case
        if balance > 1 and key > node.left.key:
            if visualize:
                self.visualize_tree(node.left, path, 
                                  [f"Left-Right case detected",
                                   f"First, perform left rotation at {node.left.key}"], 
                                  f"AVL Tree - Left-Right Case")
            node.left = self._avl_rotate_left(node.left, path, visualize)
            return self._avl_rotate_right(node, path, visualize)
        
        # Right Left Case
        if balance < -1 and key < node.right.key:
            if visualize:
                self.visualize_tree(node.right, path, 
                                  [f"Right-Left case detected",
                                   f"First, perform right rotation at {node.right.key}"], 
                                  f"AVL Tree - Right-Left Case")
            node.right = self._avl_rotate_right(node.right, path, visualize)
            return self._avl_rotate_left(node, path, visualize)
        
        return node
    
    def avl_insert(self, key):
        """Visualize the insertion process in an AVL tree."""
        self.setup_problem_folder("avl_insert_operation")
        
        # Initial tree
        self.visualize_tree(title=f"Initial AVL Tree\nInserting key: {key}")
        
        # Path for visualization
        path = []
        
        # Perform AVL insertion with visualization
        self.root = self._avl_insert(self.root, key, path, True)
        
        # Final tree
        self.visualize_tree(title=f"Final AVL Tree After Inserting {key}")
        
        return self.save_problem_data("AVL Insert", {"key": key, "result": "success"})
    
    # ===== HEAP IMPLEMENTATIONS =====
    
    def create_min_heap(self, values=None, size=None):
        """Create a min heap with the given values or random values."""
        self.ds_type = "min_heap"
        
        if values is None:
            if size is None:
                size = random.randint(5, 15)
            values = random.sample(range(1, 100), size)
            
        # Initialize heap array
        self.heap = []
        
        # Insert values one by one
        for value in values:
            self.heap_insert(value, visualize=False)
            
        return self
    
    def create_max_heap(self, values=None, size=None):
        """Create a max heap with the given values or random values."""
        self.ds_type = "max_heap"
        
        if values is None:
            if size is None:
                size = random.randint(5, 15)
            values = random.sample(range(1, 100), size)
            
        # Initialize heap array
        self.heap = []
        
        # Insert values one by one
        for value in values:
            self.heap_insert(value, visualize=False)
            
        return self
    
    def visualize_heap(self, highlighted_indices=None, operation_details=None, title=""):
        """Visualize a heap data structure."""
        if not hasattr(self, 'heap'):
            raise ValueError("Heap not initialized")
            
        if highlighted_indices is None:
            highlighted_indices = []
            
        # Color scheme definitions
        color_schemes = {
            "default": {
                "node": "lightblue",
                "highlighted": "red",
                "edge": "black",
                "text": "black"
            },
            "colorful": {
                "node": "#87CEEB",  # SkyBlue
                "highlighted": "#FF6347",  # Tomato
                "edge": "#4682B4",  # SteelBlue
                "text": "#000080"   # Navy
            },
            "monochrome": {
                "node": "#E0E0E0",
                "highlighted": "#404040",
                "edge": "#A0A0A0", 
                "text": "#000000"
            },
            "contrast": {
                "node": "#FFF0C9",
                "highlighted": "#D64550",
                "edge": "#1A1B25",
                "text": "#000000"
            }
        }
        
        # Set up figure with theme
        plt.figure(figsize=(12, 8))
        
        if self.theme == "dark":
            plt.style.use("dark_background")
            text_color = "white"
            
            # Override text colors for dark theme
            for scheme in color_schemes:
                color_schemes[scheme]["text"] = "white"
                if scheme == "monochrome":
                    color_schemes[scheme]["node"] = "#505050"
        else:
            plt.style.use("default")
            text_color = "black"
            
        # Get colors from scheme
        scheme = color_schemes.get(self.color_scheme, color_schemes["default"])
        
        # Calculate positions for heap nodes
        positions = {}
        self._calculate_heap_positions(positions)
        
        # Set axes off and title
        plt.axis('off')
        plt.title(title, fontsize=16, color=text_color)
        
        # Draw edges first
        self._draw_heap_edges(positions, scheme)
        
        # Draw nodes
        self._draw_heap_nodes(positions, scheme, highlighted_indices)
        
        # Add heap array representation
        array_text = f"Heap array: [{', '.join(map(str, self.heap))}]"
        plt.figtext(0.5, 0.95, array_text, ha='center', fontsize=12, color=text_color)
        
        # Add operation details if provided and enabled
        if operation_details and self.show_details:
            detail_text = "\n".join(operation_details)
            plt.figtext(0.5, 0.02, detail_text, ha='center', fontsize=12, 
                      color=text_color, bbox=dict(facecolor='none', edgecolor=scheme["edge"], 
                                              alpha=0.5, boxstyle='round,pad=1'))
        
        # Save the visualization
        self.save_image()
    
    def _calculate_heap_positions(self, positions):
        """Calculate positions for heap nodes."""
        if not self.heap:
            return
            
        # Get the height of the heap
        heap_height = math.floor(math.log2(len(self.heap))) + 1
        
        # Calculate positions level by level
        for i in range(len(self.heap)):
            # Calculate level and position in level
            level = math.floor(math.log2(i + 1)) + 1
            level_position = i + 1 - (2 ** (level - 1))
            positions_in_level = 2 ** (level - 1)
            
            # Calculate x coordinate
            width = 2 ** (heap_height - 1)
            x_spacing = width / positions_in_level
            x = level_position * x_spacing + x_spacing / 2 - width / 2
            
            # Calculate y coordinate (top to bottom)
            y = heap_height - level
            
            # Store position
            positions[i] = (x, y)
    
    def _draw_heap_edges(self, positions, color_scheme):
        """Draw edges between heap nodes."""
        if len(self.heap) <= 1:
            return
            
        # Edge style
        style = {
            "solid": "-",
            "dashed": "--",
            "dotted": ":",
            "dashdot": "-."
        }.get(self.edge_style, "-")
        
        # Draw edges from parent to children
        for i in range(1, len(self.heap)):
            parent_idx = (i - 1) // 2
            
            start_pos = positions[parent_idx]
            end_pos = positions[i]
            
            # Draw the edge
            plt.plot([start_pos[0], end_pos[0]], [start_pos[1], end_pos[1]], 
                   style, color=color_scheme["edge"], linewidth=1.5)
    
    def _draw_heap_nodes(self, positions, color_scheme, highlighted_indices):
        """Draw heap nodes."""
        # Node shape parameters
        node_shapes = {
            "circle": "o",
            "square": "s", 
            "triangle": "^", 
            "diamond": "d",
            "hexagon": "h"
        }
        shape = node_shapes.get(self.node_shape, "o")
        
        # Draw each node
        for i in range(len(self.heap)):
            x, y = positions[i]
            
            # Determine node color based on status
            node_color = color_scheme["highlighted"] if i in highlighted_indices else color_scheme["node"]
            
            # Draw the node
            plt.scatter(x, y, s=800, c=node_color, marker=shape, zorder=10)
            
            # Draw the node text
            plt.text(x, y, str(self.heap[i]), ha='center', va='center', fontsize=12,
                   color=color_scheme["text"], fontweight='bold', zorder=11)
    
    def heap_insert(self, value, visualize=True):
        """Insert a value into the heap and visualize the process."""
        if not hasattr(self, 'heap'):
            raise ValueError("Heap not initialized")
            
        if visualize:
            self.setup_problem_folder("heap_insert_operation")
            
            # Initial heap
            self.visualize_heap(title=f"Initial {'Min' if self.ds_type == 'min_heap' else 'Max'} Heap\nInserting value: {value}")
        
        # Insert value at the end
        self.heap.append(value)
        inserted_idx = len(self.heap) - 1
        
        if visualize:
            self.visualize_heap([inserted_idx], 
                              [f"Step 1: Insert {value} at the end of the heap"], 
                              f"Heap Insert - Added {value} at Position {inserted_idx}")
        
        # Heapify up
        current_idx = inserted_idx
        step = 2
        
        while current_idx > 0:
            parent_idx = (current_idx - 1) // 2
            
            # Check if we need to swap with parent
            need_swap = False
            
            if self.ds_type == "min_heap" and self.heap[current_idx] < self.heap[parent_idx]:
                need_swap = True
            elif self.ds_type == "max_heap" and self.heap[current_idx] > self.heap[parent_idx]:
                need_swap = True
                
            if visualize:
                self.visualize_heap([current_idx, parent_idx], 
                                  [f"Step {step}: Compare value {self.heap[current_idx]} with parent {self.heap[parent_idx]}",
                                   f"{'Swap needed' if need_swap else 'No swap needed'}"], 
                                  f"Heap Insert - Compare with Parent")
                step += 1
                
            if not need_swap:
                break
                
            # Swap with parent
            self.heap[current_idx], self.heap[parent_idx] = self.heap[parent_idx], self.heap[current_idx]
            current_idx = parent_idx
            
            if visualize:
                self.visualize_heap([current_idx], 
                                  [f"Step {step}: Swapped {self.heap[current_idx]} with its parent",
                                   f"Current position: {current_idx}"], 
                                  f"Heap Insert - Swapped with Parent")
                step += 1
        
        if visualize:
            # Final heap
            self.visualize_heap(title=f"Final {'Min' if self.ds_type == 'min_heap' else 'Max'} Heap After Inserting {value}")
            
            return self.save_problem_data("Heap Insert", {"value": value, "result": "success"})
        
        return None
    
    def heap_extract(self):
        """Extract the root (min/max) value from the heap and visualize the process."""
        if not hasattr(self, 'heap') or not self.heap:
            raise ValueError("Heap is empty or not initialized")
            
        self.setup_problem_folder("heap_extract_operation")
        
        # Initial heap
        self.visualize_heap(title=f"Initial {'Min' if self.ds_type == 'min_heap' else 'Max'} Heap\nExtracting root value")
        
        # Store root value
        root_value = self.heap[0]
        
        self.visualize_heap([0], 
                          [f"Step 1: Extract root value {root_value}"], 
                          f"Heap Extract - Extracting Root {root_value}")
        
        # If this is the last element, just remove it
        if len(self.heap) == 1:
            self.heap.pop()
            self.visualize_heap(title=f"Final {'Min' if self.ds_type == 'min_heap' else 'Max'} Heap After Extracting {root_value}")
            return self.save_problem_data("Heap Extract", {"value": root_value, "result": "success"})
        
        # Replace root with last element
        last_value = self.heap.pop()
        self.heap[0] = last_value
        
        self.visualize_heap([0], 
                          [f"Step 2: Replace root with last element {last_value}"], 
                          f"Heap Extract - Replace Root with Last Element")
        
        # Heapify down
        current_idx = 0
        step = 3
        
        while True:
            left_idx = 2 * current_idx + 1
            right_idx = 2 * current_idx + 2
            
            # Find the index to swap with
            swap_idx = current_idx
            
            if self.ds_type == "min_heap":
                # For min heap, find smallest among current, left, and right
                if left_idx < len(self.heap) and self.heap[left_idx] < self.heap[swap_idx]:
                    swap_idx = left_idx
                
                if right_idx < len(self.heap) and self.heap[right_idx] < self.heap[swap_idx]:
                    swap_idx = right_idx
            else:  # max heap
                # For max heap, find largest among current, left, and right
                if left_idx < len(self.heap) and self.heap[left_idx] > self.heap[swap_idx]:
                    swap_idx = left_idx
                
                if right_idx < len(self.heap) and self.heap[right_idx] > self.heap[swap_idx]:
                    swap_idx = right_idx
            
            highlighted = [current_idx]
            if left_idx < len(self.heap):
                highlighted.append(left_idx)
            if right_idx < len(self.heap):
                highlighted.append(right_idx)
                
            self.visualize_heap(highlighted, 
                              [f"Step {step}: Compare value {self.heap[current_idx]} with its children",
                               f"{'Left child: ' + str(self.heap[left_idx]) if left_idx < len(self.heap) else 'No left child'}",
                               f"{'Right child: ' + str(self.heap[right_idx]) if right_idx < len(self.heap) else 'No right child'}",
                               f"{'Swap needed with ' + ('left child' if swap_idx == left_idx else 'right child') if swap_idx != current_idx else 'No swap needed'}"], 
                              f"Heap Extract - Compare with Children")
            step += 1
            
            # If no swap needed, we're done
            if swap_idx == current_idx:
                break
                
            # Swap with the selected child
            self.heap[current_idx], self.heap[swap_idx] = self.heap[swap_idx], self.heap[current_idx]
            current_idx = swap_idx
            
            self.visualize_heap([current_idx], 
                              [f"Step {step}: Swapped {self.heap[current_idx]} with its parent",
                               f"Current position: {current_idx}"], 
                              f"Heap Extract - Swapped with Parent")
            step += 1
        
        # Final heap
        self.visualize_heap(title=f"Final {'Min' if self.ds_type == 'min_heap' else 'Max'} Heap After Extracting {root_value}")
        
        return self.save_problem_data("Heap Extract", {"value": root_value, "result": "success"})
    
    def heap_build(self, values):
        """Visualize the process of building a heap from an array."""
        self.setup_problem_folder("heap_build_operation")
        
        # Initialize with the array
        self.heap = list(values)
        
        # Initial unheapified array
        self.visualize_heap(title=f"Initial Array\nBuilding {'Min' if self.ds_type == 'min_heap' else 'Max'} Heap")
        
        # Heapify down starting from the last non-leaf node
        start_idx = (len(self.heap) // 2) - 1
        step = 1
        
        for i in range(start_idx, -1, -1):
            current_idx = i
            
            self.visualize_heap([current_idx], 
                              [f"Step {step}: Start heapify down from index {current_idx} (value {self.heap[current_idx]})"], 
                              f"Build Heap - Starting Heapify at Index {current_idx}")
            step += 1
            
            # Heapify down
            while True:
                left_idx = 2 * current_idx + 1
                right_idx = 2 * current_idx + 2
                
                # Find the index to swap with
                swap_idx = current_idx
                
                if self.ds_type == "min_heap":
                    # For min heap, find smallest among current, left, and right
                    if left_idx < len(self.heap) and self.heap[left_idx] < self.heap[swap_idx]:
                        swap_idx = left_idx
                    
                    if right_idx < len(self.heap) and self.heap[right_idx] < self.heap[swap_idx]:
                        swap_idx = right_idx
                else:  # max heap
                    # For max heap, find largest among current, left, and right
                    if left_idx < len(self.heap) and self.heap[left_idx] > self.heap[swap_idx]:
                        swap_idx = left_idx
                    
                    if right_idx < len(self.heap) and self.heap[right_idx] > self.heap[swap_idx]:
                        swap_idx = right_idx
                
                highlighted = [current_idx]
                if left_idx < len(self.heap):
                    highlighted.append(left_idx)
                if right_idx < len(self.heap):
                    highlighted.append(right_idx)
                    
                self.visualize_heap(highlighted, 
                                  [f"Step {step}: Compare value {self.heap[current_idx]} with its children",
                                   f"{'Left child: ' + str(self.heap[left_idx]) if left_idx < len(self.heap) else 'No left child'}",
                                   f"{'Right child: ' + str(self.heap[right_idx]) if right_idx < len(self.heap) else 'No right child'}",
                                   f"{'Swap needed with ' + ('left child' if swap_idx == left_idx else 'right child') if swap_idx != current_idx else 'No swap needed'}"], 
                                  f"Build Heap - Compare with Children")
                step += 1
                
                # If no swap needed, we're done with this node
                if swap_idx == current_idx:
                    break
                    
                # Swap with the selected child
                self.heap[current_idx], self.heap[swap_idx] = self.heap[swap_idx], self.heap[current_idx]
                current_idx = swap_idx
                
                self.visualize_heap([current_idx], 
                                  [f"Step {step}: Swapped {self.heap[current_idx]} with its parent",
                                   f"Current position: {current_idx}"], 
                                  f"Build Heap - Swapped with Parent")
                step += 1
        
        # Final heap
        self.visualize_heap(title=f"Final {'Min' if self.ds_type == 'min_heap' else 'Max'} Heap After Building")
        
        return self.save_problem_data("Heap Build", {"initial_values": values, "result": "success"})

def main():
    """Demonstrate the data structure visualizer."""
    # Example usage for Binary Search Tree
    visualizer = DataStructureVisualizer()
    
    # Create and visualize a BST
    visualizer.create_bst([50, 30, 70, 20, 40, 60, 80])
    visualizer.set_color_scheme("colorful").set_node_shape("circle").set_theme("light")
    visualizer.bst_insert(45)
    
    # Create and visualize an AVL Tree
    visualizer = DataStructureVisualizer()
    visualizer.create_avl_tree([10, 20, 30, 40, 50, 25])
    visualizer.set_color_scheme("contrast").set_node_shape("diamond").set_theme("dark")
    visualizer.avl_insert(35)
    
    # Create and visualize a Min Heap
    visualizer = DataStructureVisualizer()
    visualizer.create_min_heap([10, 20, 15, 30, 40])
    visualizer.set_color_scheme("monochrome").set_node_shape("triangle")
    visualizer.heap_insert(5)
    visualizer.heap_extract()
    
    print("Visualization demo completed.")

if __name__ == "__main__":
    main()
import os
import random
from create_data_structure import DataStructureVisualizer

def generate_bst_examples(output_dir="bst_examples"):
    """Generate examples of Binary Search Tree operations with diverse visualizations."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Visual style combinations
    styles = [
        {"color_scheme": "default", "node_shape": "circle", "edge_style": "solid", "theme": "light"},
        {"color_scheme": "colorful", "node_shape": "diamond", "edge_style": "solid", "theme": "dark"},
        {"color_scheme": "monochrome", "node_shape": "square", "edge_style": "dashed", "theme": "light"},
        {"color_scheme": "contrast", "node_shape": "triangle", "edge_style": "dotted", "theme": "dark"}
    ]
    
    # BST operations to visualize
    operations = [
        {"name": "Insert", "func": lambda v, key: v.bst_insert(key)},
        {"name": "Search", "func": lambda v, key: v.bst_search(key)},
        {"name": "Delete", "func": lambda v, key: v.bst_delete(key)},
        {"name": "Inorder Traversal", "func": lambda v, _: v.bst_traversal("inorder")},
        {"name": "Preorder Traversal", "func": lambda v, _: v.bst_traversal("preorder")},
        {"name": "Postorder Traversal", "func": lambda v, _: v.bst_traversal("postorder")},
        {"name": "Level Order Traversal", "func": lambda v, _: v.bst_traversal("levelorder")}
    ]
    
    print(f"Generating BST examples in {output_dir}...")
    
    # Generate examples for each style and operation
    for style in styles:
        style_name = f"{style['color_scheme']}_{style['node_shape']}_{style['theme']}"
        print(f"\nVisual style: {style_name}")
        
        for op in operations:
            print(f"  Operation: {op['name']}")
            
            # Create a new visualizer with the given style
            visualizer = DataStructureVisualizer(output_dir=output_dir)
            visualizer.set_color_scheme(style['color_scheme'])
            visualizer.set_node_shape(style['node_shape'])
            visualizer.set_edge_style(style['edge_style'])
            visualizer.set_theme(style['theme'])
            
            # Create a random BST
            keys = random.sample(range(1, 100), random.randint(7, 12))
            visualizer.create_bst(keys)
            
            # Select a key for operations that need one
            if op['name'] in ["Insert", "Search", "Delete"]:
                if op['name'] == "Insert":
                    # For insert, use a new key
                    target_key = random.randint(1, 100)
                    while target_key in keys:
                        target_key = random.randint(1, 100)
                else:
                    # For search/delete, use an existing key
                    target_key = random.choice(keys)
                    
                print(f"    Key: {target_key}")
                op['func'](visualizer, target_key)
            else:
                # For traversals, no key needed
                op['func'](visualizer, None)
                
            print(f"    Output saved to: {visualizer.problem_folder}")

def generate_avl_examples(output_dir="avl_examples"):
    """Generate examples of AVL Tree operations with diverse visualizations."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Visual style combinations
    styles = [
        {"color_scheme": "default", "node_shape": "circle", "edge_style": "solid", "theme": "light"},
        {"color_scheme": "colorful", "node_shape": "hexagon", "edge_style": "solid", "theme": "dark"},
        {"color_scheme": "contrast", "node_shape": "triangle", "edge_style": "dotted", "theme": "light"}
    ]
    
    print(f"Generating AVL Tree examples in {output_dir}...")
    
    # Generate examples for each style
    for style in styles:
        style_name = f"{style['color_scheme']}_{style['node_shape']}_{style['theme']}"
        print(f"\nVisual style: {style_name}")
        
        # Create a new visualizer with the given style
        visualizer = DataStructureVisualizer(output_dir=output_dir)
        visualizer.set_color_scheme(style['color_scheme'])
        visualizer.set_node_shape(style['node_shape'])
        visualizer.set_edge_style(style['edge_style'])
        visualizer.set_theme(style['theme'])
        
        # Create a random AVL tree
        keys = random.sample(range(1, 100), random.randint(6, 10))
        visualizer.create_avl_tree(keys)
        
        # Perform AVL insert operation
        insert_key = random.randint(1, 100)
        while insert_key in keys:
            insert_key = random.randint(1, 100)
            
        print(f"  Inserting key: {insert_key}")
        visualizer.avl_insert(insert_key)
        
        print(f"  Output saved to: {visualizer.problem_folder}")

def generate_heap_examples(output_dir="heap_examples"):
    """Generate examples of Heap operations with diverse visualizations."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Visual style combinations
    styles = [
        {"color_scheme": "default", "node_shape": "circle", "edge_style": "solid", "theme": "light"},
        {"color_scheme": "colorful", "node_shape": "diamond", "edge_style": "solid", "theme": "dark"}
    ]
    
    # Heap operations to visualize
    operations = [
        {"name": "Insert", "func": lambda v, val: v.heap_insert(val)},
        {"name": "Extract", "func": lambda v, _: v.heap_extract()},
        {"name": "Build", "func": lambda v, vals: v.heap_build(vals)}
    ]
    
    # Heap types
    heap_types = [
        {"name": "Min Heap", "func": lambda v, vals: v.create_min_heap(vals)},
        {"name": "Max Heap", "func": lambda v, vals: v.create_max_heap(vals)}
    ]
    
    print(f"Generating Heap examples in {output_dir}...")
    
    # Generate examples for each style, heap type, and operation
    for style in styles:
        style_name = f"{style['color_scheme']}_{style['node_shape']}_{style['theme']}"
        print(f"\nVisual style: {style_name}")
        
        for heap_type in heap_types:
            print(f"  Heap type: {heap_type['name']}")
            
            for op in operations:
                print(f"    Operation: {op['name']}")
                
                # Create a new visualizer with the given style
                visualizer = DataStructureVisualizer(output_dir=output_dir)
                visualizer.set_color_scheme(style['color_scheme'])
                visualizer.set_node_shape(style['node_shape'])
                visualizer.set_edge_style(style['edge_style'])
                visualizer.set_theme(style['theme'])
                
                # Generate random values
                values = random.sample(range(1, 100), random.randint(6, 10))
                
                if op['name'] == "Build":
                    # For build operation, just create the empty heap structure
                    if heap_type['name'] == "Min Heap":
                        visualizer.ds_type = "min_heap"
                    else:
                        visualizer.ds_type = "max_heap"
                    
                    print(f"      Values: {values}")
                    op['func'](visualizer, values)
                else:
                    # For other operations, first create a heap
                    heap_type['func'](visualizer, values)
                    
                    if op['name'] == "Insert":
                        # For insert, generate a new value
                        insert_val = random.randint(1, 100)
                        print(f"      Inserting value: {insert_val}")
                        op['func'](visualizer, insert_val)
                    else:
                        # For extract, no value needed
                        op['func'](visualizer, None)
                        
                print(f"      Output saved to: {visualizer.problem_folder}")

def generate_specific_examples(output_dir="specific_examples"):
    """Generate specific, carefully crafted examples."""
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Generating specific examples in {output_dir}...")
    
    # Example 1: BST with all traversals in colorful style
    print("\nExample 1: BST with all traversals")
    visualizer = DataStructureVisualizer(output_dir=output_dir)
    visualizer.set_color_scheme("colorful").set_node_shape("circle").set_theme("dark")
    
    # Create a balanced BST
    visualizer.create_bst([50, 25, 75, 12, 37, 62, 87])
    
    # Perform all traversals
    for traversal in ["inorder", "preorder", "postorder", "levelorder"]:
        visualizer.bst_traversal(traversal)
        print(f"  {traversal.capitalize()} traversal saved to: {visualizer.problem_folder}")
    
    # Example 2: AVL Tree insertion with rotations
    print("\nExample 2: AVL Tree with rotations")
    visualizer = DataStructureVisualizer(output_dir=output_dir)
    visualizer.set_color_scheme("contrast").set_node_shape("triangle").set_theme("light")
    
    # Create an AVL tree that will need rotations
    visualizer.create_avl_tree([10, 20, 30, 40, 50])
    
    # Insert value that causes rotation
    visualizer.avl_insert(25)
    print(f"  AVL insertion with rotation saved to: {visualizer.problem_folder}")
    
    # Example 3: BST deletion with all cases
    print("\nExample 3: BST deletion cases")
    
    # Case 1: Delete leaf node
    visualizer = DataStructureVisualizer(output_dir=output_dir)
    visualizer.set_color_scheme("default").set_node_shape("square").set_theme("light")
    visualizer.create_bst([50, 30, 70, 20, 40, 60, 80])
    visualizer.bst_delete(20)  # Delete leaf
    print(f"  Leaf node deletion saved to: {visualizer.problem_folder}")
    
    # Case 2: Delete node with one child
    visualizer = DataStructureVisualizer(output_dir=output_dir)
    visualizer.set_color_scheme("monochrome").set_node_shape("diamond").set_theme("dark")
    visualizer.create_bst([50, 30, 70, 20, 40, 60, 80, 65])
    visualizer.bst_delete(60)  # Delete node with one child
    print(f"  One-child node deletion saved to: {visualizer.problem_folder}")
    
    # Case 3: Delete node with two children
    visualizer = DataStructureVisualizer(output_dir=output_dir)
    visualizer.set_color_scheme("colorful").set_node_shape("hexagon").set_theme("light")
    visualizer.create_bst([50, 30, 70, 20, 40, 60, 80, 35, 45])
    visualizer.bst_delete(30)  # Delete node with two children
    print(f"  Two-children node deletion saved to: {visualizer.problem_folder}")
    
    # Example 4: Min Heap operations
    print("\nExample 4: Min Heap operations")
    visualizer = DataStructureVisualizer(output_dir=output_dir)
    visualizer.set_color_scheme("colorful").set_node_shape("circle").set_theme("dark")
    
    # Build a min heap from array
    values = [30, 20, 10, 15, 5, 7, 3]
    visualizer.ds_type = "min_heap"
    visualizer.heap_build(values)
    print(f"  Min heap building saved to: {visualizer.problem_folder}")
    
    # Example 5: Max Heap operations
    print("\nExample 5: Max Heap operations")
    visualizer = DataStructureVisualizer(output_dir=output_dir)
    visualizer.set_color_scheme("contrast").set_node_shape("triangle").set_theme("light")
    
    # Create a max heap
    visualizer.create_max_heap([10, 20, 30, 40, 50])
    
    # Insert value
    visualizer.heap_insert(45)
    print(f"  Max heap insertion saved to: {visualizer.problem_folder}")
    
    # Extract max
    visualizer = DataStructureVisualizer(output_dir=output_dir)
    visualizer.set_color_scheme("monochrome").set_node_shape("square").set_theme("dark")
    visualizer.create_max_heap([50, 40, 30, 20, 10, 25, 15])
    visualizer.heap_extract()
    print(f"  Max heap extraction saved to: {visualizer.problem_folder}")

if __name__ == "__main__":
    # Generate examples for all data structures
    generate_bst_examples()
    generate_avl_examples()
    generate_heap_examples()
    generate_specific_examples()
    
    print("\nAll examples generated successfully!")
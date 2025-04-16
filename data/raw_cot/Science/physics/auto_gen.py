import os
import random
from physics_simulator import PhysicsSimulator

def generate_projectile_examples(output_dir="projectile_examples"):
    """Generate examples of projectile motion with diverse visualizations."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Visual style combinations
    styles = [
        {"color_scheme": "default", "theme": "light", "line_style": "solid", "marker_style": "circle"},
        {"color_scheme": "colorful", "theme": "dark", "line_style": "solid", "marker_style": "triangle"},
        {"color_scheme": "monochrome", "theme": "light", "line_style": "dashed", "marker_style": "square"},
        {"color_scheme": "contrast", "theme": "dark", "line_style": "dotted", "marker_style": "diamond"}
    ]
    
    # Physical parameter combinations
    physics_params = [
        {"environment": "earth", "air_resistance": False, "initial_height": 0},
        {"environment": "moon", "air_resistance": False, "initial_height": 0},
        {"environment": "earth", "air_resistance": True, "initial_height": 0},
        {"environment": "mars", "air_resistance": True, "initial_height": 10}
    ]
    
    print(f"Generating projectile motion examples in {output_dir}...")
    
    # Generate examples for each style and physics combination
    for i, style in enumerate(styles):
        style_name = f"{style['color_scheme']}_{style['theme']}"
        print(f"\nVisual style {i+1}/{len(styles)}: {style_name}")
        
        for j, params in enumerate(physics_params):
            env_text = params["environment"]
            air_text = "with_air" if params["air_resistance"] else "no_air"
            params_name = f"{env_text}_{air_text}_h{params['initial_height']}"
            
            print(f"  Physics setup {j+1}/{len(physics_params)}: {params_name}")
            
            # Create new simulator with the given style
            simulator = PhysicsSimulator(output_dir=output_dir)
            simulator.set_color_scheme(style["color_scheme"])
            simulator.set_theme(style["theme"])
            simulator.set_line_style(style["line_style"])
            simulator.set_marker_style(style["marker_style"])
            
            # Set physics environment
            simulator.set_environment(params["environment"])
            
            if params["air_resistance"]:
                simulator.set_air_resistance(True, drag_coefficient=0.1)
            
            # Generate random projectile parameters
            initial_velocity = random.uniform(30, 50)
            launch_angle = random.uniform(30, 60)
            
            # Setup and run simulation
            simulator.setup_projectile_motion(
                initial_velocity=initial_velocity,
                launch_angle=launch_angle,
                initial_height=params["initial_height"]
            )
            
            print(f"    Initial velocity: {initial_velocity:.2f} m/s at {launch_angle:.2f}°")
            simulator.simulate_projectile_motion()
            
            print(f"    Output saved to: {simulator.problem_folder}")

def generate_environment_comparison_examples(output_dir="environment_examples"):
    """Generate examples comparing projectile motion in different environments."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Visual styles for each comparison
    styles = [
        {"color_scheme": "colorful", "theme": "light"},
        {"color_scheme": "contrast", "theme": "dark"}
    ]
    
    # Different environment comparisons
    comparisons = [
        ["earth", "moon", "mars"],
        ["earth", "moon"],
        ["earth", "mars"],
        ["moon", "mars"]
    ]
    
    print(f"Generating environment comparison examples in {output_dir}...")
    
    for i, style in enumerate(styles):
        style_name = f"{style['color_scheme']}_{style['theme']}"
        print(f"\nVisual style {i+1}/{len(styles)}: {style_name}")
        
        for j, environments in enumerate(comparisons):
            env_text = "_".join(environments)
            print(f"  Comparison {j+1}/{len(comparisons)}: {env_text}")
            
            # Create new simulator with the given style
            simulator = PhysicsSimulator(output_dir=output_dir)
            simulator.set_color_scheme(style["color_scheme"])
            simulator.set_theme(style["theme"])
            
            # Generate random projectile parameters
            initial_velocity = random.uniform(30, 50)
            launch_angle = random.uniform(30, 60)
            
            # Setup projectile motion (uses earth gravity by default)
            simulator.setup_projectile_motion(
                initial_velocity=initial_velocity,
                launch_angle=launch_angle
            )
            
            # Generate the environment comparison visualization
            simulator.setup_problem_folder(f"environment_comparison_{env_text}")
            simulator.visualize_environment_comparison(environments)
            
            # Save problem data
            result = {
                "initial_conditions": {
                    "initial_velocity": initial_velocity,
                    "launch_angle": launch_angle,
                    "environments": environments
                }
            }
            simulator.save_problem_data("Environment Comparison", result)
            
            print(f"    Output saved to: {simulator.problem_folder}")

def generate_specific_examples(output_dir="specific_examples"):
    """Generate specific, carefully crafted physics examples."""
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Generating specific examples in {output_dir}...")
    
    # Example 1: Earth standard projectile with colorful visualization
    print("\nExample 1: Standard Earth projectile")
    simulator = PhysicsSimulator(output_dir=output_dir)
    simulator.set_color_scheme("colorful").set_theme("light")
    simulator.setup_projectile_motion(initial_velocity=45, launch_angle=45)
    simulator.simulate_projectile_motion()
    print(f"  Output saved to: {simulator.problem_folder}")
    
    # Example 2: Moon projectile with high initial velocity
    print("\nExample 2: Moon projectile with high velocity")
    simulator = PhysicsSimulator(output_dir=output_dir)
    simulator.set_color_scheme("contrast").set_theme("dark")
    simulator.set_environment("moon")
    simulator.setup_projectile_motion(initial_velocity=60, launch_angle=30)
    simulator.simulate_projectile_motion()
    print(f"  Output saved to: {simulator.problem_folder}")
    
    # Example 3: Earth projectile with air resistance
    print("\nExample 3: Earth projectile with air resistance")
    simulator = PhysicsSimulator(output_dir=output_dir)
    simulator.set_color_scheme("default").set_theme("light")
    simulator.set_air_resistance(True, drag_coefficient=0.2)
    simulator.setup_projectile_motion(initial_velocity=50, launch_angle=60)
    simulator.simulate_projectile_motion()
    print(f"  Output saved to: {simulator.problem_folder}")
    
    # Example 4: Mars projectile from elevation
    print("\nExample 4: Mars projectile from elevation")
    simulator = PhysicsSimulator(output_dir=output_dir)
    simulator.set_color_scheme("monochrome").set_theme("dark")
    simulator.set_environment("mars")
    simulator.setup_projectile_motion(initial_velocity=40, launch_angle=40, initial_height=20)
    simulator.simulate_projectile_motion()
    print(f"  Output saved to: {simulator.problem_folder}")
    
    # Example 5: Custom gravity environment
    print("\nExample 5: Custom gravity environment")
    simulator = PhysicsSimulator(output_dir=output_dir)
    simulator.set_color_scheme("colorful").set_theme("dark")
    simulator.set_custom_gravity(5.0)  # Custom gravity between Earth and Mars
    simulator.setup_projectile_motion(initial_velocity=55, launch_angle=50)
    simulator.simulate_projectile_motion()
    print(f"  Output saved to: {simulator.problem_folder}")

def generate_air_resistance_comparison(output_dir="air_resistance_examples"):
    """Generate examples comparing projectile motion with and without air resistance."""
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Generating air resistance comparison examples in {output_dir}...")
    
    # Generate comparisons with different initial conditions
    velocities = [30, 50, 70]
    angles = [30, 45, 60]
    
    for velocity in velocities:
        for angle in angles:
            print(f"\nComparing air resistance effects: v₀ = {velocity} m/s, θ = {angle}°")
            
            # Create simulator for comparison
            simulator = PhysicsSimulator(output_dir=output_dir)
            simulator.set_color_scheme("contrast").set_theme("light")
            
            # Setup basic projectile motion
            simulator.setup_projectile_motion(
                initial_velocity=velocity,
                launch_angle=angle
            )
            
            # Store original trajectory (no air resistance)
            no_air_x = simulator.x.copy()
            no_air_y = simulator.y.copy()
            no_air_range = simulator.horizontal_range
            no_air_max_height = simulator.max_height
            
            # Set air resistance and recalculate
            simulator.set_air_resistance(True, drag_coefficient=0.2)
            simulator._simulate_projectile_motion()
            
            # Setup plot
            simulator.setup_problem_folder(f"air_resistance_v{velocity}_angle{angle}")
            ax = simulator.setup_plot(
                title=f"Air Resistance Comparison (v₀ = {velocity} m/s, θ = {angle}°)",
                xlabel="Distance (m)",
                ylabel="Height (m)"
            )
            
            # Get current styles
            styles = simulator.get_plot_styles()
            colors = styles["colors"]
            ls = styles["line_style"]
            
            # Plot both trajectories
            ax.plot(no_air_x, no_air_y, color=colors["trajectory"], linestyle=ls,
                  label="No Air Resistance")
            ax.plot(simulator.x, simulator.y, color=colors["velocity"], linestyle=ls,
                  label="With Air Resistance")
            
            # Mark key points
            ax.plot(no_air_range, 0, marker="o", markersize=6, color=colors["trajectory"])
            ax.plot(simulator.horizontal_range, 0, marker="o", markersize=6, color=colors["velocity"])
            
            # Add annotations if enabled
            if simulator.show_annotations:
                # Range difference
                range_diff = no_air_range - simulator.horizontal_range
                range_diff_percent = (range_diff / no_air_range) * 100
                
                # Height difference
                height_diff = no_air_max_height - simulator.max_height
                height_diff_percent = (height_diff / no_air_max_height) * 100
                
                # Add text annotations
                ax.text(0.05, 0.95, 
                      f"Range reduction: {range_diff:.2f} m ({range_diff_percent:.1f}%)",
                      transform=ax.transAxes, fontsize=10, verticalalignment='top',
                      color=colors["annotations"])
                      
                ax.text(0.05, 0.9, 
                      f"Max height reduction: {height_diff:.2f} m ({height_diff_percent:.1f}%)",
                      transform=ax.transAxes, fontsize=10, verticalalignment='top',
                      color=colors["annotations"])
            
            # Add legend
            ax.legend(loc="upper right", facecolor=colors["figure"], 
                   edgecolor=colors["grid"], labelcolor=colors["text"])
            
            # Set axis limits
            max_x = max(no_air_range, simulator.horizontal_range) * 1.1
            max_y = max(no_air_max_height, simulator.max_height) * 1.2
            ax.set_xlim(-max_x * 0.05, max_x)
            ax.set_ylim(-max_y * 0.05, max_y)
            
            # Save image
            simulator.save_image(filename="air_resistance_comparison.png")
            
            # Save problem data
            result = {
                "initial_conditions": {
                    "initial_velocity": velocity,
                    "launch_angle": angle
                },
                "no_air_resistance": {
                    "horizontal_range": no_air_range,
                    "max_height": no_air_max_height
                },
                "with_air_resistance": {
                    "horizontal_range": simulator.horizontal_range,
                    "max_height": simulator.max_height,
                    "drag_coefficient": simulator.drag_coefficient
                },
                "comparison": {
                    "range_reduction": range_diff,
                    "range_reduction_percent": range_diff_percent,
                    "height_reduction": height_diff,
                    "height_reduction_percent": height_diff_percent
                }
            }
            
            simulator.save_problem_data("Air Resistance Comparison", result)
            print(f"  Output saved to: {simulator.problem_folder}")

if __name__ == "__main__":
    # Generate examples for physics simulations
    generate_projectile_examples()
    generate_environment_comparison_examples()
    generate_specific_examples()
    generate_air_resistance_comparison()
    
    print("\nAll physics examples generated successfully!")
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.colors as mcolors
from mpl_toolkits.mplot3d import Axes3D
import os
import json
from datetime import datetime
import random

class PhysicsSimulator:
    def __init__(self, output_dir="physics_outputs"):
        """Initialize the physics simulator with visualization options."""
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Create timestamp for unique folder name
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Store visualization data
        self.images = []
        self.problem_folder = None
        self.physics_type = "unknown"
        
        # Enhanced visualization options
        self.color_scheme = "default"  # default, colorful, monochrome, contrast
        self.theme = "light"           # light, dark
        self.line_style = "solid"      # solid, dashed, dotted, dashdot
        self.marker_style = "circle"   # circle, square, triangle, diamond, plus
        self.show_annotations = True   # Show additional text annotations
        self.show_grid = True          # Show grid lines
        self.show_forces = True        # Show force vectors
        self.frame_rate = 60           # Animation frame rate
        
        # Physics environment options
        self.environment = "earth"     # earth, moon, mars, custom
        self.air_resistance = False    # Consider air resistance
        self.drag_coefficient = 0.1    # Drag coefficient if air resistance is enabled
        
        # Gravity constants for different environments (m/s²)
        self.gravity_constants = {
            "earth": 9.8,
            "moon": 1.62,
            "mars": 3.72,
            "custom": 9.8  # Default custom value
        }
        
        # Set current gravity
        self.gravity = self.gravity_constants[self.environment]
        
    # ===== VISUALIZATION SETTING METHODS =====
    
    def set_color_scheme(self, scheme):
        """Set the color scheme for visualization."""
        valid_schemes = ["default", "colorful", "monochrome", "contrast"]
        if scheme not in valid_schemes:
            raise ValueError(f"Invalid color scheme. Choose from: {valid_schemes}")
        self.color_scheme = scheme
        return self
        
    def set_theme(self, theme):
        """Set the visualization theme."""
        valid_themes = ["light", "dark"]
        if theme not in valid_themes:
            raise ValueError(f"Invalid theme. Choose from: {valid_themes}")
        self.theme = theme
        return self
        
    def set_line_style(self, style):
        """Set the line style for plots."""
        valid_styles = ["solid", "dashed", "dotted", "dashdot"]
        if style not in valid_styles:
            raise ValueError(f"Invalid line style. Choose from: {valid_styles}")
        self.line_style = style
        return self
        
    def set_marker_style(self, style):
        """Set the marker style for plots."""
        valid_styles = ["circle", "square", "triangle", "diamond", "plus", "none"]
        if style not in valid_styles:
            raise ValueError(f"Invalid marker style. Choose from: {valid_styles}")
        self.marker_style = style
        return self
        
    def set_show_annotations(self, enabled=True):
        """Set whether to show text annotations."""
        self.show_annotations = enabled
        return self
        
    def set_show_grid(self, enabled=True):
        """Set whether to show grid lines."""
        self.show_grid = enabled
        return self
        
    def set_show_forces(self, enabled=True):
        """Set whether to show force vectors."""
        self.show_forces = enabled
        return self
        
    def set_frame_rate(self, rate=60):
        """Set animation frame rate."""
        if rate < 1:
            raise ValueError("Frame rate must be at least 1 fps")
        self.frame_rate = rate
        return self
        
    # ===== PHYSICS ENVIRONMENT METHODS =====
    
    def set_environment(self, env):
        """Set the physics environment (affects gravity)."""
        valid_envs = list(self.gravity_constants.keys())
        if env not in valid_envs:
            raise ValueError(f"Invalid environment. Choose from: {valid_envs}")
        self.environment = env
        self.gravity = self.gravity_constants[env]
        return self
        
    def set_custom_gravity(self, g):
        """Set a custom gravity value."""
        if g < 0:
            raise ValueError("Gravity must be non-negative")
        self.environment = "custom"
        self.gravity = g
        self.gravity_constants["custom"] = g
        return self
        
    def set_air_resistance(self, enabled=True, drag_coefficient=0.1):
        """Enable or disable air resistance."""
        self.air_resistance = enabled
        if drag_coefficient <= 0 or drag_coefficient > 1:
            raise ValueError("Drag coefficient must be between 0 and 1")
        self.drag_coefficient = drag_coefficient
        return self
    
    # ===== UTILITY METHODS =====
    
    def setup_problem_folder(self, problem_name):
        """Create a unique folder for the current problem."""
        self.problem_folder = os.path.join(
            self.output_dir, 
            f"{self.physics_type}_{problem_name}_{self.timestamp}"
        )
        os.makedirs(self.problem_folder, exist_ok=True)
        self.images = []
        return self.problem_folder
        
    def save_image(self, title="", filename=None):
        """Save the current figure to the problem folder."""
        if not self.problem_folder:
            raise ValueError("Problem folder must be set up first.")
            
        if filename is None:
            filename = f"step{len(self.images)+1}.png"
            
        fname = os.path.join(self.problem_folder, filename)
        plt.savefig(fname, bbox_inches='tight')
        plt.close()
        self.images.append(fname)
        return fname
        
    def save_problem_data(self, operation, result, additional_info=None):
        """Save problem data to JSON."""
        if not self.problem_folder:
            return None
            
        # Basic problem description
        problem_desc = f"Simulate {operation} with the given parameters."
        
        # Save problem data
        problem_data = {
            "problem": problem_desc,
            "images": [os.path.basename(img) for img in self.images],
            "solution": result,
            "visualization": {
                "color_scheme": self.color_scheme,
                "theme": self.theme,
                "line_style": self.line_style,
                "marker_style": self.marker_style,
                "show_annotations": self.show_annotations,
                "show_grid": self.show_grid,
                "show_forces": self.show_forces
            },
            "environment": {
                "type": self.environment,
                "gravity": self.gravity,
                "air_resistance": self.air_resistance,
                "drag_coefficient": self.drag_coefficient if self.air_resistance else None
            }
        }
        
        # Add additional info if provided
        if additional_info:
            problem_data["additional_info"] = additional_info
        
        # Save problem data to JSON file
        with open(os.path.join(self.problem_folder, "problem_data.json"), "w") as f:
            json.dump(problem_data, f, indent=2)
        
        return problem_data
    
    def get_plot_styles(self):
        """Get the current plot styles based on color scheme and theme."""
        # Color schemes
        color_schemes = {
            "default": {
                "light": {
                    "trajectory": "blue",
                    "velocity": "red",
                    "acceleration": "green",
                    "markers": "purple",
                    "annotations": "black",
                    "grid": "#cccccc",
                    "figure": "white",
                    "text": "black"
                },
                "dark": {
                    "trajectory": "#00aaff",
                    "velocity": "#ff5555",
                    "acceleration": "#55ff55",
                    "markers": "#ff55ff",
                    "annotations": "white",
                    "grid": "#555555",
                    "figure": "#222222",
                    "text": "white"
                }
            },
            "colorful": {
                "light": {
                    "trajectory": "#3498db",
                    "velocity": "#e74c3c",
                    "acceleration": "#2ecc71",
                    "markers": "#9b59b6",
                    "annotations": "#2c3e50",
                    "grid": "#bdc3c7",
                    "figure": "#ecf0f1",
                    "text": "#2c3e50"
                },
                "dark": {
                    "trajectory": "#3498db",
                    "velocity": "#e74c3c",
                    "acceleration": "#2ecc71",
                    "markers": "#9b59b6",
                    "annotations": "#ecf0f1",
                    "grid": "#7f8c8d",
                    "figure": "#2c3e50",
                    "text": "#ecf0f1"
                }
            },
            "monochrome": {
                "light": {
                    "trajectory": "#000000",
                    "velocity": "#333333",
                    "acceleration": "#666666",
                    "markers": "#999999",
                    "annotations": "#000000",
                    "grid": "#cccccc",
                    "figure": "#ffffff",
                    "text": "#000000"
                },
                "dark": {
                    "trajectory": "#ffffff",
                    "velocity": "#cccccc",
                    "acceleration": "#999999",
                    "markers": "#666666",
                    "annotations": "#ffffff",
                    "grid": "#333333",
                    "figure": "#000000",
                    "text": "#ffffff"
                }
            },
            "contrast": {
                "light": {
                    "trajectory": "#0000ff",
                    "velocity": "#ff0000",
                    "acceleration": "#00aa00",
                    "markers": "#aa00aa",
                    "annotations": "#000000",
                    "grid": "#aaaaaa",
                    "figure": "#ffffff",
                    "text": "#000000"
                },
                "dark": {
                    "trajectory": "#00ff00",
                    "velocity": "#ff0000",
                    "acceleration": "#00ffff",
                    "markers": "#ffff00",
                    "annotations": "#ffffff",
                    "grid": "#555555",
                    "figure": "#000000",
                    "text": "#ffffff"
                }
            }
        }
        
        # Line styles
        line_styles = {
            "solid": "-",
            "dashed": "--",
            "dotted": ":",
            "dashdot": "-."
        }
        
        # Marker styles
        marker_styles = {
            "circle": "o",
            "square": "s",
            "triangle": "^",
            "diamond": "d",
            "plus": "+",
            "none": None
        }
        
        # Get the appropriate colors for the current theme and color scheme
        colors = color_schemes.get(self.color_scheme, color_schemes["default"]).get(self.theme, color_schemes["default"]["light"])
        
        # Get the current line and marker styles
        ls = line_styles.get(self.line_style, "-")
        marker = marker_styles.get(self.marker_style, "o")
        
        return {
            "colors": colors,
            "line_style": ls,
            "marker": marker
        }
    
    def setup_plot(self, figsize=(10, 8), title="", xlabel="", ylabel=""):
        """Set up a plot with the current theme and styles."""
        plt.figure(figsize=figsize)
        
        # Get current styles
        styles = self.get_plot_styles()
        colors = styles["colors"]
        
        # Set background and text colors based on theme
        if self.theme == "dark":
            plt.style.use("dark_background")
            plt.rcParams.update({
                'figure.facecolor': colors["figure"],
                'axes.facecolor': colors["figure"],
                'text.color': colors["text"],
                'axes.labelcolor': colors["text"],
                'xtick.color': colors["text"],
                'ytick.color': colors["text"]
            })
        else:
            plt.style.use("default")
            plt.rcParams.update({
                'figure.facecolor': colors["figure"],
                'axes.facecolor': colors["figure"],
                'text.color': colors["text"],
                'axes.labelcolor': colors["text"],
                'xtick.color': colors["text"],
                'ytick.color': colors["text"]
            })
        
        # Add grid if enabled
        if self.show_grid:
            plt.grid(True, color=colors["grid"], linestyle='-', linewidth=0.5, alpha=0.7)
        
        # Set title and labels
        plt.title(title, fontsize=14, color=colors["text"])
        plt.xlabel(xlabel, fontsize=12, color=colors["text"])
        plt.ylabel(ylabel, fontsize=12, color=colors["text"])
        
        return plt.gca()  # Return the current axis
    
    # ===== PROJECTILE MOTION SIMULATION =====
    
    def setup_projectile_motion(self, initial_velocity, launch_angle, initial_height=0, 
                               mass=1.0, projectile_radius=0.1, time_step=0.01):
        """Set up projectile motion simulation parameters."""
        self.physics_type = "projectile_motion"
        
        # Store basic parameters
        self.initial_velocity = initial_velocity
        self.launch_angle = launch_angle
        self.initial_height = initial_height
        self.mass = mass
        self.projectile_radius = projectile_radius
        self.time_step = time_step
        
        # Convert angle to radians
        angle_rad = np.radians(launch_angle)
        
        # Calculate initial velocity components
        self.v0x = initial_velocity * np.cos(angle_rad)
        self.v0y = initial_velocity * np.sin(angle_rad)
        
        # Calculate time of flight (without air resistance)
        discriminant = self.v0y**2 + 2 * self.gravity * initial_height
        if discriminant < 0:
            raise ValueError("Invalid parameters: projectile would never reach the ground")
            
        self.time_of_flight = (self.v0y + np.sqrt(discriminant)) / self.gravity
        
        # Calculate maximum height
        self.max_height = initial_height + self.v0y**2 / (2 * self.gravity)
        
        # Calculate range (without air resistance)
        self.horizontal_range = self.v0x * self.time_of_flight
        
        # Create time array for simulation
        self.t = np.arange(0, self.time_of_flight + time_step, time_step)
        
        # Initialize position and velocity arrays
        self.x = np.zeros_like(self.t)
        self.y = np.zeros_like(self.t)
        self.vx = np.zeros_like(self.t)
        self.vy = np.zeros_like(self.t)
        
        # Run simulation
        self._simulate_projectile_motion()
        
        return self
    
    def _simulate_projectile_motion(self):
        """Run the projectile motion simulation with or without air resistance."""
        # Initialize with initial conditions
        self.x[0] = 0
        self.y[0] = self.initial_height
        self.vx[0] = self.v0x
        self.vy[0] = self.v0y
        
        if not self.air_resistance:
            # Analytical solution without air resistance
            for i in range(1, len(self.t)):
                t_val = self.t[i]
                self.x[i] = self.v0x * t_val
                self.y[i] = self.initial_height + self.v0y * t_val - 0.5 * self.gravity * t_val**2
                self.vx[i] = self.v0x
                self.vy[i] = self.v0y - self.gravity * t_val
                
                # Stop if projectile hits the ground
                if self.y[i] < 0:
                    self.y[i] = 0
                    self.t = self.t[:i+1]
                    self.x = self.x[:i+1]
                    self.y = self.y[:i+1]
                    self.vx = self.vx[:i+1]
                    self.vy = self.vy[:i+1]
                    break
        else:
            # Numerical solution with air resistance
            # F_drag = -0.5 * rho * C_d * A * v^2 * v_hat
            # For simplicity, we'll use a lumped parameter: drag_coefficient
            
            for i in range(1, len(self.t)):
                # Current velocity
                vx_current = self.vx[i-1]
                vy_current = self.vy[i-1]
                v_magnitude = np.sqrt(vx_current**2 + vy_current**2)
                
                # Drag force components (normalized by mass)
                drag_factor = self.drag_coefficient * v_magnitude
                ax_drag = -drag_factor * vx_current
                ay_drag = -drag_factor * vy_current
                
                # Total acceleration
                ax = ax_drag
                ay = -self.gravity + ay_drag
                
                # Update velocity (Euler integration)
                self.vx[i] = vx_current + ax * self.time_step
                self.vy[i] = vy_current + ay * self.time_step
                
                # Update position
                self.x[i] = self.x[i-1] + self.vx[i] * self.time_step
                self.y[i] = self.y[i-1] + self.vy[i] * self.time_step
                
                # Stop if projectile hits the ground
                if self.y[i] < 0:
                    self.y[i] = 0
                    self.t = self.t[:i+1]
                    self.x = self.x[:i+1]
                    self.y = self.y[:i+1]
                    self.vx = self.vx[:i+1]
                    self.vy = self.vy[:i+1]
                    break
                    
        # Update actual time of flight and range based on simulation
        self.time_of_flight = self.t[-1]
        self.horizontal_range = self.x[-1]
        
        # Find actual maximum height from simulation data
        self.max_height = np.max(self.y)
        
    def visualize_projectile_initial_conditions(self):
        """Visualize the initial conditions of projectile motion."""
        # Set up plot
        ax = self.setup_plot(
            title="Projectile Motion - Initial Conditions",
            xlabel="Distance (m)",
            ylabel="Height (m)"
        )
        
        # Get current styles
        styles = self.get_plot_styles()
        colors = styles["colors"]
        ls = styles["line_style"]
        marker = styles["marker"]
        
        # Plot trajectory preview with low opacity
        ax.plot(self.x, self.y, color=colors["trajectory"], linestyle=ls, alpha=0.3)
        
        # Plot initial velocity vector
        arrow_scale = self.initial_velocity / 10
        ax.arrow(0, self.initial_height, 
                arrow_scale * np.cos(np.radians(self.launch_angle)), 
                arrow_scale * np.sin(np.radians(self.launch_angle)),
                head_width=arrow_scale/2, head_length=arrow_scale/2, 
                fc=colors["velocity"], ec=colors["velocity"])
        
        # Add annotations if enabled
        if self.show_annotations:
            env_text = f"Environment: {self.environment.capitalize()} (g = {self.gravity:.2f} m/s²)"
            init_text = f"Initial velocity: {self.initial_velocity:.2f} m/s at {self.launch_angle:.1f}°"
            res_text = f"Air resistance: {'Enabled' if self.air_resistance else 'Disabled'}"
            
            ax.text(0.05, 0.95, env_text, transform=ax.transAxes, fontsize=10, 
                   verticalalignment='top', color=colors["annotations"])
            ax.text(0.05, 0.9, init_text, transform=ax.transAxes, fontsize=10, 
                   verticalalignment='top', color=colors["annotations"])
            ax.text(0.05, 0.85, res_text, transform=ax.transAxes, fontsize=10, 
                   verticalalignment='top', color=colors["annotations"])
        
        # Set axis limits to show the full trajectory with some margin
        max_x = self.horizontal_range * 1.1
        max_y = self.max_height * 1.2
        ax.set_xlim(-max_x * 0.05, max_x)
        ax.set_ylim(-max_y * 0.05, max_y)
        
        # Save image
        return self.save_image(filename="initial_conditions.png")
    
    def visualize_projectile_velocity_components(self):
        """Visualize the velocity components of projectile motion."""
        # Set up plot
        ax = self.setup_plot(
            title="Projectile Motion - Velocity Components",
            xlabel="Distance (m)",
            ylabel="Height (m)"
        )
        
        # Get current styles
        styles = self.get_plot_styles()
        colors = styles["colors"]
        
        # Calculate arrow scales for better visualization
        arrow_scale = self.initial_velocity / 10
        
        # Plot velocity components
        ax.arrow(0, self.initial_height, arrow_scale * np.cos(np.radians(self.launch_angle)), 0,
               head_width=arrow_scale/3, head_length=arrow_scale/3, 
               fc=colors["trajectory"], ec=colors["trajectory"])
               
        ax.arrow(0, self.initial_height, 0, arrow_scale * np.sin(np.radians(self.launch_angle)),
               head_width=arrow_scale/3, head_length=arrow_scale/3, 
               fc=colors["acceleration"], ec=colors["acceleration"])
               
        ax.arrow(0, self.initial_height, 
                arrow_scale * np.cos(np.radians(self.launch_angle)), 
                arrow_scale * np.sin(np.radians(self.launch_angle)),
                head_width=arrow_scale/3, head_length=arrow_scale/3, 
                fc=colors["velocity"], ec=colors["velocity"])
        
        # Add annotations if enabled
        if self.show_annotations:
            ax.text(arrow_scale * np.cos(np.radians(self.launch_angle)) / 2, 
                   self.initial_height - arrow_scale/3,
                   f"v₀ₓ = {self.v0x:.2f} m/s", fontsize=10, 
                   color=colors["trajectory"], ha='center')
                   
            ax.text(-arrow_scale/3, 
                   self.initial_height + arrow_scale * np.sin(np.radians(self.launch_angle)) / 2,
                   f"v₀ᵧ = {self.v0y:.2f} m/s", fontsize=10, 
                   color=colors["acceleration"], ha='right', va='center')
                   
            ax.text(arrow_scale * np.cos(np.radians(self.launch_angle)) / 2,
                   self.initial_height + arrow_scale * np.sin(np.radians(self.launch_angle)) / 2,
                   f"|v₀| = {self.initial_velocity:.2f} m/s", fontsize=10,
                   color=colors["velocity"], ha='left', va='bottom')
        
        # Set axis limits
        ax.set_xlim(-arrow_scale, arrow_scale * 1.5)
        ax.set_ylim(self.initial_height - arrow_scale, 
                  self.initial_height + arrow_scale * 1.5)
        
        # Save image
        return self.save_image(filename="velocity_components.png")
    
    def visualize_projectile_trajectory(self):
        """Visualize the complete trajectory of projectile motion with key points."""
        # Set up plot
        ax = self.setup_plot(
            title="Projectile Motion - Complete Trajectory",
            xlabel="Distance (m)",
            ylabel="Height (m)"
        )
        
        # Get current styles
        styles = self.get_plot_styles()
        colors = styles["colors"]
        ls = styles["line_style"]
        marker = styles["marker"]
        
        # Plot trajectory
        ax.plot(self.x, self.y, color=colors["trajectory"], linestyle=ls, linewidth=2)
        
        # Mark key points
        # Starting point
        ax.plot(0, self.initial_height, marker=marker, markersize=8, 
               color=colors["markers"])
        
        if self.show_annotations:
            ax.text(5, self.initial_height + 2, "Start", fontsize=10,
                  color=colors["annotations"])
        
        # Maximum height point
        # Find index of maximum height
        max_height_idx = np.argmax(self.y)
        max_height_x = self.x[max_height_idx]
        
        ax.plot(max_height_x, self.max_height, marker=marker, markersize=8,
               color=colors["markers"])
               
        if self.show_annotations:
            ax.text(max_height_x + self.horizontal_range * 0.05, self.max_height,
                  f"Max Height: {self.max_height:.2f} m", fontsize=10,
                  color=colors["annotations"])
        
        # End point
        ax.plot(self.horizontal_range, 0, marker=marker, markersize=8,
               color=colors["markers"])
               
        if self.show_annotations:
            ax.text(self.horizontal_range - self.horizontal_range * 0.2, 2,
                  f"Range: {self.horizontal_range:.2f} m", fontsize=10,
                  color=colors["annotations"])
        
        # Add velocity vectors at different points if enabled
        if self.show_forces:
            # Number of vectors to show
            num_vectors = 5
            vector_indices = np.linspace(0, len(self.t) - 1, num_vectors, dtype=int)
            
            for idx in vector_indices:
                x_val = self.x[idx]
                y_val = self.y[idx]
                vx = self.vx[idx]
                vy = self.vy[idx]
                
                # Scale vectors for visibility
                v_scale = self.initial_velocity / 15
                scaled_vx = vx * v_scale / self.initial_velocity
                scaled_vy = vy * v_scale / self.initial_velocity
                
                ax.arrow(x_val, y_val, scaled_vx, scaled_vy,
                       head_width=v_scale/3, head_length=v_scale/3,
                       fc=colors["velocity"], ec=colors["velocity"], alpha=0.7)
        
        # Set axis limits to show the full trajectory with some margin
        max_x = self.horizontal_range * 1.1
        max_y = self.max_height * 1.2
        ax.set_xlim(-max_x * 0.05, max_x)
        ax.set_ylim(-max_y * 0.05, max_y)
        
        # Save image
        return self.save_image(filename="trajectory.png")
    
    def visualize_projectile_kinematics(self):
        """Visualize position, velocity, and acceleration as functions of time."""
        # Create figure with 3 subplots
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
        
        # Get current styles
        styles = self.get_plot_styles()
        colors = styles["colors"]
        ls = styles["line_style"]
        
        # Set background color based on theme
        if self.theme == "dark":
            fig.patch.set_facecolor(colors["figure"])
            for ax in [ax1, ax2, ax3]:
                ax.set_facecolor(colors["figure"])
                ax.tick_params(colors=colors["text"])
        
        # Position plot
        ax1.plot(self.t, self.x, color=colors["trajectory"], linestyle=ls, label='x-position')
        ax1.plot(self.t, self.y, color=colors["velocity"], linestyle=ls, label='y-position')
        ax1.set_ylabel('Position (m)', color=colors["text"])
        ax1.set_title('Position vs Time', color=colors["text"])
        ax1.legend(facecolor=colors["figure"], edgecolor=colors["grid"], labelcolor=colors["text"])
        if self.show_grid:
            ax1.grid(True, color=colors["grid"], linestyle='-', linewidth=0.5, alpha=0.7)
        
        # Velocity plot
        ax2.plot(self.t, self.vx, color=colors["trajectory"], linestyle=ls, label='x-velocity')
        ax2.plot(self.t, self.vy, color=colors["velocity"], linestyle=ls, label='y-velocity')
        ax2.set_ylabel('Velocity (m/s)', color=colors["text"])
        ax2.set_title('Velocity vs Time', color=colors["text"])
        ax2.legend(facecolor=colors["figure"], edgecolor=colors["grid"], labelcolor=colors["text"])
        if self.show_grid:
            ax2.grid(True, color=colors["grid"], linestyle='-', linewidth=0.5, alpha=0.7)
        
        # Acceleration plot
        if not self.air_resistance:
            # Without air resistance, acceleration is constant
            ax3.plot(self.t, np.zeros_like(self.t), color=colors["trajectory"], 
                   linestyle=ls, label='x-acceleration')
            ax3.plot(self.t, -self.gravity * np.ones_like(self.t), color=colors["velocity"], 
                   linestyle=ls, label='y-acceleration')
        else:
            # With air resistance, calculate acceleration from velocity changes
            ax_air = np.zeros_like(self.t)
            ay_air = np.zeros_like(self.t)
            
            # Approximate acceleration using central differences
            for i in range(1, len(self.t)-1):
                ax_air[i] = (self.vx[i+1] - self.vx[i-1]) / (self.t[i+1] - self.t[i-1])
                ay_air[i] = (self.vy[i+1] - self.vy[i-1]) / (self.t[i+1] - self.t[i-1])
            
            # Forward difference for first point
            ax_air[0] = (self.vx[1] - self.vx[0]) / (self.t[1] - self.t[0])
            ay_air[0] = (self.vy[1] - self.vy[0]) / (self.t[1] - self.t[0])
            
            # Backward difference for last point
            ax_air[-1] = (self.vx[-1] - self.vx[-2]) / (self.t[-1] - self.t[-2])
            ay_air[-1] = (self.vy[-1] - self.vy[-2]) / (self.t[-1] - self.t[-2])
            
            ax3.plot(self.t, ax_air, color=colors["trajectory"], linestyle=ls, label='x-acceleration')
            ax3.plot(self.t, ay_air, color=colors["velocity"], linestyle=ls, label='y-acceleration')
        
        ax3.set_xlabel('Time (s)', color=colors["text"])
        ax3.set_ylabel('Acceleration (m/s²)', color=colors["text"])
        ax3.set_title('Acceleration vs Time', color=colors["text"])
        ax3.legend(facecolor=colors["figure"], edgecolor=colors["grid"], labelcolor=colors["text"])
        if self.show_grid:
            ax3.grid(True, color=colors["grid"], linestyle='-', linewidth=0.5, alpha=0.7)
        
        # Add annotations if enabled
        if self.show_annotations:
            plt.figtext(0.5, 0.01, 
                      f"Environment: {self.environment.capitalize()} (g = {self.gravity:.2f} m/s²) | " +
                      f"Air resistance: {'Enabled' if self.air_resistance else 'Disabled'}",
                      ha="center", fontsize=10, color=colors["text"])
        
        plt.tight_layout()
        
        # Save image
        return self.save_image(filename="kinematics.png")
    
    def visualize_projectile_energy(self):
        """Visualize the energy components during projectile motion."""
        # Set up plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
        
        # Get current styles
        styles = self.get_plot_styles()
        colors = styles["colors"]
        ls = styles["line_style"]
        
        # Set background color based on theme
        if self.theme == "dark":
            fig.patch.set_facecolor(colors["figure"])
            for ax in [ax1, ax2]:
                ax.set_facecolor(colors["figure"])
                ax.tick_params(colors=colors["text"])
        
        # Calculate energy components
        # Kinetic energy: KE = 0.5 * m * v²
        KE = 0.5 * self.mass * (self.vx**2 + self.vy**2)
        
        # Potential energy: PE = m * g * h
        PE = self.mass * self.gravity * self.y
        
        # Total energy
        TE = KE + PE
        
        # Energy plot
        ax1.plot(self.t, KE, color=colors["velocity"], linestyle=ls, label='Kinetic Energy')
        ax1.plot(self.t, PE, color=colors["acceleration"], linestyle=ls, label='Potential Energy')
        ax1.plot(self.t, TE, color=colors["trajectory"], linestyle=ls, label='Total Energy')
        ax1.set_ylabel('Energy (J)', color=colors["text"])
        ax1.set_title('Energy vs Time', color=colors["text"])
        ax1.legend(facecolor=colors["figure"], edgecolor=colors["grid"], labelcolor=colors["text"])
        if self.show_grid:
            ax1.grid(True, color=colors["grid"], linestyle='-', linewidth=0.5, alpha=0.7)
        
        # Energy percentage plot
        total_energy_initial = TE[0]
        ax2.plot(self.t, KE/total_energy_initial*100, color=colors["velocity"], 
               linestyle=ls, label='Kinetic Energy %')
        ax2.plot(self.t, PE/total_energy_initial*100, color=colors["acceleration"], 
               linestyle=ls, label='Potential Energy %')
        ax2.plot(self.t, TE/total_energy_initial*100, color=colors["trajectory"], 
               linestyle=ls, label='Total Energy %')
        ax2.set_xlabel('Time (s)', color=colors["text"])
        ax2.set_ylabel('Energy (%)', color=colors["text"])
        ax2.set_title('Energy Percentage vs Time', color=colors["text"])
        ax2.legend(facecolor=colors["figure"], edgecolor=colors["grid"], labelcolor=colors["text"])
        if self.show_grid:
            ax2.grid(True, color=colors["grid"], linestyle='-', linewidth=0.5, alpha=0.7)
            
        # Show conservation of energy or loss due to air resistance
        if self.show_annotations:
            if self.air_resistance:
                energy_loss = (TE[0] - TE[-1]) / TE[0] * 100
                plt.figtext(0.5, 0.01, 
                          f"Energy loss due to air resistance: {energy_loss:.2f}%",
                          ha="center", fontsize=10, color=colors["text"])
            else:
                plt.figtext(0.5, 0.01, 
                          "Total energy is conserved (no energy loss)",
                          ha="center", fontsize=10, color=colors["text"])
        
        plt.tight_layout()
        
        # Save image
        return self.save_image(filename="energy.png")
    
    def visualize_projectile_3d(self):
        """Create a 3D visualization of projectile motion."""
        # Create 3D figure
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Get current styles
        styles = self.get_plot_styles()
        colors = styles["colors"]
        ls = styles["line_style"]
        
        # Set background color based on theme
        if self.theme == "dark":
            fig.patch.set_facecolor(colors["figure"])
            ax.set_facecolor(colors["figure"])
            ax.tick_params(colors=colors["text"])
        
        # For 3D visualization, add a z-dimension (constant for regular projectile motion)
        z = np.zeros_like(self.x)
        
        # Plot trajectory in 3D
        ax.plot(self.x, z, self.y, color=colors["trajectory"], linestyle=ls, linewidth=2)
        
        # Plot shadow/projection on the ground
        ax.plot(self.x, z, np.zeros_like(self.y), color=colors["annotations"], 
              linestyle='--', alpha=0.5)
        
        # Mark key points
        # Starting point
        ax.scatter(0, 0, self.initial_height, color=colors["markers"], s=100)
        
        # Maximum height point
        max_height_idx = np.argmax(self.y)
        max_height_x = self.x[max_height_idx]
        ax.scatter(max_height_x, 0, self.max_height, color=colors["markers"], s=100)
        
        # End point
        ax.scatter(self.horizontal_range, 0, 0, color=colors["markers"], s=100)
        
        # Add annotations if enabled
        if self.show_annotations:
            ax.text(0, 0, self.initial_height + 2, "Start", color=colors["text"], fontsize=10)
            ax.text(max_height_x, 0, self.max_height + 2, 
                  f"Max Height: {self.max_height:.2f} m", color=colors["text"], fontsize=10)
            ax.text(self.horizontal_range, 0, 2, 
                  f"Range: {self.horizontal_range:.2f} m", color=colors["text"], fontsize=10)
        
        # Add velocity vectors if enabled
        if self.show_forces:
            # Number of vectors to show
            num_vectors = 5
            vector_indices = np.linspace(0, len(self.t) - 1, num_vectors, dtype=int)
            
            for idx in vector_indices:
                x_val = self.x[idx]
                y_val = self.y[idx]
                vx = self.vx[idx]
                vy = self.vy[idx]
                
                # Scale vectors for visibility
                v_scale = self.initial_velocity / 15
                scaled_vx = vx * v_scale / self.initial_velocity
                scaled_vy = vy * v_scale / self.initial_velocity
                
                # Draw velocity vector
                ax.quiver(x_val, 0, y_val, scaled_vx, 0, scaled_vy, 
                        color=colors["velocity"], arrow_length_ratio=0.2, alpha=0.7)
        
        # Set labels and title
        ax.set_xlabel('Distance (m)', color=colors["text"])
        ax.set_ylabel('Z (m)', color=colors["text"])
        ax.set_zlabel('Height (m)', color=colors["text"])
        ax.set_title('Projectile Motion - 3D Visualization', color=colors["text"])
        
        # Add grid
        if self.show_grid:
            ax.grid(True, color=colors["grid"], linestyle='-', linewidth=0.5, alpha=0.7)
        
        # Set axis limits
        max_x = self.horizontal_range * 1.1
        max_y = 10  # Arbitrary z-axis depth
        max_z = self.max_height * 1.2
        ax.set_xlim(0, max_x)
        ax.set_ylim(-max_y/2, max_y/2)
        ax.set_zlim(0, max_z)
        
        # Save image
        return self.save_image(filename="trajectory_3d.png")
    
    def visualize_environment_comparison(self, environments=None):
        """Compare projectile motion in different environments."""
        if environments is None:
            environments = ["earth", "moon", "mars"]
            
        # Validate environments
        for env in environments:
            if env not in self.gravity_constants:
                raise ValueError(f"Invalid environment: {env}")
        
        # Set up plot
        ax = self.setup_plot(
            title="Projectile Motion - Environment Comparison",
            xlabel="Distance (m)",
            ylabel="Height (m)"
        )
        
        # Get current styles
        styles = self.get_plot_styles()
        colors = styles["colors"]
        ls = styles["line_style"]
        
        # Save original environment settings
        original_environment = self.environment
        original_gravity = self.gravity
        original_x = self.x.copy()
        original_y = self.y.copy()
        original_range = self.horizontal_range
        original_max_height = self.max_height
        
        # Color map for different environments
        env_colors = {
            "earth": colors["trajectory"],
            "moon": colors["velocity"],
            "mars": colors["acceleration"],
            "custom": colors["markers"]
        }
        
        # Plot trajectories for each environment
        for env in environments:
            # Set environment and recalculate
            self.set_environment(env)
            self._simulate_projectile_motion()
            
            # Plot trajectory
            ax.plot(self.x, self.y, label=f"{env.capitalize()} (g = {self.gravity:.2f} m/s²)",
                  color=env_colors.get(env, colors["annotations"]), linestyle=ls)
            
            # Mark key points
            max_height_idx = np.argmax(self.y)
            max_height_x = self.x[max_height_idx]
            
            # Maximum height point
            ax.plot(max_height_x, self.max_height, marker="o", markersize=6,
                  color=env_colors.get(env, colors["annotations"]))
            
            # End point
            ax.plot(self.horizontal_range, 0, marker="o", markersize=6,
                  color=env_colors.get(env, colors["annotations"]))
            
            # Add annotations if enabled
            if self.show_annotations:
                ax.text(self.horizontal_range, 2, 
                      f"{env.capitalize()}: {self.horizontal_range:.1f} m", 
                      fontsize=8, color=env_colors.get(env, colors["annotations"]))
        
        # Add legend
        ax.legend(loc="upper right", facecolor=colors["figure"], 
               edgecolor=colors["grid"], labelcolor=colors["text"])
        
        # Set axis limits to show all trajectories
        max_range = max(self.horizontal_range, original_range)
        max_height = max(self.max_height, original_max_height)
        ax.set_xlim(-max_range * 0.05, max_range * 1.1)
        ax.set_ylim(-max_height * 0.05, max_height * 1.2)
        
        # Add additional information if enabled
        if self.show_annotations:
            details = [
                f"Initial velocity: {self.initial_velocity:.2f} m/s at {self.launch_angle:.1f}°",
                f"Initial height: {self.initial_height:.2f} m",
                f"Air resistance: {'Enabled' if self.air_resistance else 'Disabled'}"
            ]
            
            ax.text(0.05, 0.95, "\n".join(details), transform=ax.transAxes, fontsize=9,
                  verticalalignment='top', color=colors["annotations"],
                  bbox=dict(facecolor=colors["figure"], edgecolor=colors["grid"], alpha=0.7))
        
        # Restore original environment settings
        self.environment = original_environment
        self.gravity = original_gravity
        self.x = original_x
        self.y = original_y
        self.horizontal_range = original_range
        self.max_height = original_max_height
        
        # Save image
        return self.save_image(filename="environment_comparison.png")
    
    def simulate_projectile_motion(self):
        """Run a complete projectile motion simulation with all visualizations."""
        if not hasattr(self, 'initial_velocity'):
            raise ValueError("Projectile motion parameters not set. Call setup_projectile_motion first.")
            
        self.setup_problem_folder("projectile_motion")
        
        # Create all visualizations
        self.visualize_projectile_initial_conditions()
        self.visualize_projectile_velocity_components()
        self.visualize_projectile_trajectory()
        self.visualize_projectile_kinematics()
        self.visualize_projectile_energy()
        self.visualize_projectile_3d()
        self.visualize_environment_comparison()
        
        # Generate problem description
        env_text = f"on {self.environment}" if self.environment != "earth" else ""
        air_text = " with air resistance" if self.air_resistance else ""
        
        problem_desc = (
            f"A projectile is launched {env_text}{air_text} from an initial height of "
            f"{self.initial_height:.1f} m with an initial velocity of {self.initial_velocity:.1f} m/s "
            f"at an angle of {self.launch_angle:.1f}° above the horizontal. "
            f"Find the maximum height reached, the time of flight, and the horizontal range of the projectile."
        )
        
        # Prepare results
        result = {
            "initial_conditions": {
                "initial_velocity": self.initial_velocity,
                "launch_angle": self.launch_angle,
                "initial_height": self.initial_height,
                "environment": self.environment,
                "gravity": self.gravity,
                "air_resistance": self.air_resistance,
                "drag_coefficient": self.drag_coefficient if self.air_resistance else None
            },
            "results": {
                "max_height": self.max_height,
                "time_of_flight": self.time_of_flight,
                "horizontal_range": self.horizontal_range,
                "energy_conservation": not self.air_resistance
            }
        }
        
        # Save full problem data
        return self.save_problem_data("Projectile Motion", result, {"problem_description": problem_desc})

def main():
    """Demonstrate the physics simulator."""
    # Example usage for projectile motion
    simulator = PhysicsSimulator()
    
    # Create and visualize projectile motion with default Earth settings
    simulator.setup_projectile_motion(initial_velocity=40, launch_angle=45)
    simulator.set_color_scheme("colorful").set_theme("dark").set_line_style("solid")
    simulator.simulate_projectile_motion()
    
    # Create projectile motion with air resistance and moon gravity
    simulator = PhysicsSimulator()
    simulator.set_environment("moon")
    simulator.set_air_resistance(True, drag_coefficient=0.1)
    simulator.setup_projectile_motion(initial_velocity=30, launch_angle=60, initial_height=10)
    simulator.set_color_scheme("contrast").set_theme("light").set_marker_style("triangle")
    simulator.simulate_projectile_motion()
    
    print("Physics simulation demo completed. Check the output directory for results.")

if __name__ == "__main__":
    main()
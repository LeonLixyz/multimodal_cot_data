# VLM Reasoning

## Repository Structure

We have two main components:

- **raw_cot/**: Raw implementations of various reasoning tasks. We will include all the code for generating raw CoT
  traces here. 
  - [**graph/**](raw_cot/graph/README.md): Graph algorithm visualizations (BFS, DFS, Dijkstra, etc.)
  - [**2d_image_inference/**](raw_cot/2d_image_inference/README.md): Image inference tasks
  - **VisualSketchpad/**: VisualSketchpad code
  - **physics/**: Physics simulation visualizations
  - **data_structure/**: Data structure visualizations
  - **3d_reasoning/**: Under development. We have a unity version, exploring blender version. 
  - **geometry/**: Under development. We have the part from visual sketchpad. 
  - **visual_reasoning/**: Under development. We have the output from the visual sketchpad like visual search.
  - **math/**: Under development.
  - **chemistry/**: Under development.
  - **mazes/**: Under development. 
  - **chess/**: Under development.
  - **games/**: Under development. include tasks like tetris, sudoku, etc.

- **parse_mmcot/**: Tools for parsing and analyzing multimodal chain-of-thought reasoning
  - **parsed_cot/**: Parsed chain-of-thought data
  - **visual_sketchpad/**: Currently developed a pipeline for parsing the VisualSketchpad data into MM-CoT traces.
  - **python_data/**: Under development. Need different prompt.


## TODO

- add code for each of the subtasks in the raw_cot folder.
- cluster the tasks by the four categories: Science, 2d Visual Reasoning, Strategic Games, and 3D Visual Reasoning.
- we probaly need to customize the prompts for each of the subtasks as well.


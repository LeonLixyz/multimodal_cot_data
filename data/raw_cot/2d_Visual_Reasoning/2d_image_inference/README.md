# 2D Image Inference Puzzle Generator

A tool for generating visual reasoning puzzles that test 2D image understanding and spatial reasoning capabilities.

## Puzzle Types

- **Jigsaw**: jigsaw pieces
- **Multi-piece**: Grid-based puzzles with multiple pieces removed, default 2 pieces removed
- **Fractal Cutout**: Complex shapes with various boundary types (TODO: most does not work)

## Difficulty Levels

TODO: need to tune this better. 

- Easy
- Medium
- Hard

## Usage

see `gen.py` for more details. 

```
python gen.py --image path/to/image.jpg --output_dir outputs
```

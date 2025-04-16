# VLM Reasoning

## Repository Structure

Current repository is organized as follows:

- **data**: 
  - **raw_cot**: Raw CoT traces
    - **2d_Visual_Reasoning**: 
      - [**2d_image_inference**](data/raw_cot/2d_Visual_Reasoning/2d_image_inference/README.md) draw cutlines, randomize the contours, many different cropping and let the model see
        which one is different. <span style="color:red">TODO</span>: some cut code is not correct.
      - **VisualSketchpad** visual search task.
      - **Visual Search** <span style="color:red">TODO</span>: more generic visual search code without text reasoning traces, such as drawing bounding boxes.
    - **3d_Physical_Visual_Reasoning**: 
      - **3d_reasoning** currently only has primitive 3d shapes. <span style="color:red">TODO</span>: adding different
        shapes such as pokemon, vehicle, etc.
      - **VLA_data** <span style="color:red">TODO</span>: add VLA data.
      - **physics_simulator** <span style="color:red">TODO</span>: add physics simulator code.
    - **Science**: 
      - **data_structure** Support head and tree.
      - **geometry** With data from visual sketchpad, and some scraped from internet textbook. <span style="color:red">TODO</span>: add graph data.
      - [**graph**](data/raw_cot/Science/graph/README.md) <span style="color:red">TODO</span>: check if the results are correct.
      - **math** <span style="color:red">TODO</span>: add math section.
      - **physics** Currently manually scraped from internet textbook. <span style="color:red">TODO</span>: add physics
        data (most likely we have to manually collect the data, I included a prompt in the data folder).
    - **Strategic_Games**: 
      - **chess**
      - **maze**
      - **sudoko**

  - **parse_mmcot**: parsing raw data into actual reasoning traces with GPT.


- **model**: 

  Qwen 2.5 VL has the best performance on various tasks such as MMMU and MathVista. It was also trained on interleaved
  text and image data, so should already have good unified representation. Let's use this model first. Another choice
  can be Gemma 3, but not as good as Qwen 2.5 VL.

  We should also refer back to the metamorph code.
  - **Qwen2.5-VL**: 
    - `qwen_a2a.py`: My very primitive approach to tune the model into a any to any task, only for understanding
      purposes. see also the jupyter notebook `qwen.ipynb` for a demo. <span style="color:red">TODO</span>: the
      complexity lies in the image resolution and positional embedding.




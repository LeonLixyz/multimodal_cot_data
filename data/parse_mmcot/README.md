# OpenAI API Parsing

## Setup
export your OPENAI_API_KEY first:

```bash
export OPENAI_API_KEY="your-key-here"
```

We will have two steps. The first step is to set up the input data.

## Prepare input

### Command line to generate the input for API call

```bash
python get_api_input.py --input_dir raw_data/ --prompt_file prompt.txt --output_folder output/
```

### The format of the raw_data
The format of the raw_data should be like this:
```
.
├── images
│   ├── problem_image_1.jpg
│   ├── problem_image_2.jpg
│   ├── ...
│   └── reasoning_image_1.jpg
│   ├── reasoning_image_2.jpg
│   └── ...
└── the raw reasoning text file
```

The command line above will also save the raw input in the `output_folder` folder so we can better later replace the
images with the placeholder.

### Raw reasoning text file
The raw reasoning text file should be the reasoning trace either you generated or you obtained somewhere else. The key
is that all the images should be replaced with placeholder `[problem_image_1]`, `[reasoning_image_1]`, etc.

### Prompt file
See [system](./prompts/system.txt) as an example. I tried a bunch of prompts and this one works the best. You can add in
more guidelines to help the model to generate a better trace. 

An important thing to note is that when we ask the model to generate the trace, we ask it to use the image placeholder we
provided as well, in contrast to generate any form of real images (e.g. base64 encoded images). This helps to enhance
the quality of the generated trace.

For more details on how we prepare the input for the API call, you can refer to the
[get_api_input.py](./get_api_input.py) script to see how we are loading in the image for the API call.


### 2. Call API

The command line to call the API is as follows:

```bash
python call_api.py --input_folder output/ --temperature 0.7 --model gpt-4.1
```
We only need to give the model the previous folder where we created the input for the API call. The output will be saved
to `output/api_output.json`.


## Batch processing

TODO: implement batch processing.

## Parse OpenAI output into huggingface dataset

TODO: implement the script to parse the OpenAI output into a huggingface dataset.



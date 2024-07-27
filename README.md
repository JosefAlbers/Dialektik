# Dialektik

Merge. Synthesize. Create. Dialektik generates new content by fusing ideas from diverse sources, revealing unexpected insights and perspectives.

## Features

- Loads and processes datasets from multiple sources
- Summarizes text into concise bullet points
- Synthesizes bullet points into detailed articles
- Supports various AI models for text generation
- Model-agnostic design allows easy swapping of different LLMs

## Requirements

- Required: `datasets`, `huggingface_hub`
- Optional: `phi-3-vision-mlx` *(required only if you need to create a new dataset with the provided `setup()` function for custom dataset processing)*

## Installation

To install Dialektik with core dependencies only:

   ```
   pip install dialektik
   ```

To install Dialektik with all dependencies, including those required for the setup() function:

   ```
   pip install dialektik[setup]
   ```

Note: Install the full version if you plan to process custom datasets using the `setup()` function.

## Setup

1. Clone the repository:

   ```
   git clone https://github.com/JosefAlbersç/Dialektik.git
   cd Dialektik
   ```

2. Install the required dependencies:

   ```
   pip install -r requirements.txt
   ```

## Usage

## Command Line Interface

Dialektik can be used from the command line after installation. Here are some example usages:

1. Generate a synthesis with default settings:

   ```
   dialektik
   ```

2. Specify sources:

   ```
   dialektik --source arxiv
   ```

3. Set the number of bullet points per book and choose a different model:

   ```
   dialektik --per-book 5 --model "your-preferred-model"
   ```

4. Run the setup function:

   ```
   dialektik --setup
   ```

5. For a full list of options, use:

    ```
    dialektik --help
    ```

### Accessing the Dataset

**Important Note**: The default dataset at 'JosefAlbers/StampyAI-alignment-research-dataset' is currently being prepared (ETA: 18 hours). Please check back later if unavailable.

The default dataset is to be publicly available. You do not need to set up any environment variables or run the setup() function to use `dialektik` with this dataset.

### Synthesizing content

To generate a synthesis, simply run:

   ```python
   from dialektik import synthesize

   output = synthesize()
   ```

You can customize the synthesis process by passing optional parameters:

   ```python
   output = synthesize(
      list_source=['your_source'],
      per_book=3,
      api_model="mistralai/Mistral-Nemo-Instruct-2407"
   )
   ```

### (Optional) Using Custom Datasets

If you want to use your own dataset:

1. Prepare your dataset according to the required format.
2. Modify the `PATH_DS` variable in the code to point to your dataset.
3. If your dataset is private or requires authentication, set up the following environment variables:
   - `HF_WRITE_TOKEN`: Hugging Face write token (for pushing datasets)
   - `HF_READ_TOKEN`: Hugging Face read token (for accessing private datasets)

Note: The `setup()` function provided in the code is a demonstration of how you might process a custom dataset. Different datasets may require different processing steps, so you'll need to adapt this function to your specific needs.

## Customizing the LLM

Dialektik is designed to be model-agnostic. To use a different language model:

1. Simply pass the name of your chosen model to the `synthesize()` function using the `api_model` parameter.
2. Modify the `mistral_api()` function or create a new function that interfaces with your chosen LLM.
3. Update the `synthesize()` function to use your new LLM interface.

The default model is "mistralai/Mistral-Nemo-Instruct-2407", but you can easily change this by passing a different `api_model` parameter to the `synthesize()` function.

## License

This project is licensed under the [MIT License](LICENSE).

## Citation

<a href="https://zenodo.org/doi/10.5281/zenodo.11403221"><img src="https://zenodo.org/badge/806709541.svg" alt="DOI"></a>
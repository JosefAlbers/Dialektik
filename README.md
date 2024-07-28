# Dialektik

Merge. Synthesize. Create. Dialektik generates new content by fusing ideas from diverse sources, revealing unexpected insights and perspectives through a dialectical process.

## Features

- Loads and processes datasets from multiple sources
- Summarizes text into concise bullet points
- Generates thesis, antithesis, and synthesis from summarized content
- Supports various AI models for text generation
- Model-agnostic design allows easy swapping of different LLMs

## Requirements

- Required: `datasets`, `huggingface_hub`
- Optional: `phi-3-vision-mlx` (required only if you need to create a new dataset with the provided `setup()` function for custom dataset processing)

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

## Usage

### Command Line Interface

Dialektik can be used from the command line after installation. Here are some example usages:

1. Generate a synthesis with default settings:

   ```
   dialektik
   ```

2. Specify sources:

   ```
   dialektik --source arxiv
   ```

3. Set the number of books, bullet points per book, and choose a different model:

   ```
   dialektik --num-book 5 --per-book 4 --model "your-preferred-model"
   ```

4. Run the setup function:

   ```
   dialektik --setup
   ```

5. For a full list of options, use:

   ```
   dialektik --help
   ```

### Python API

You can also use Dialektik in your Python scripts:

```python
from dialektik import synthesize

# Generate a synthesis with default settings
thesis, antithesis, synthesis = synthesize()

# Customize the synthesis process
output = synthesize(
   list_source=['your_source'],
   num_book=3,
   per_book=3,
   api_model="mistralai/Mistral-Nemo-Instruct-2407"
)
```

### Accessing the Dataset

The default dataset at 'JosefAlbers/StampyAI-alignment-research-dataset' is publicly available. You don't need to set up any environment variables or run the setup() function to use `dialektik` with this dataset.

### (Optional) Using Custom Datasets

If you want to use your own dataset:

1. Prepare your dataset according to the required format.
2. Modify the `PATH_DS` variable in the code to point to your dataset.
3. If your dataset is private or requires authentication, set up the following environment variables:
   - `HF_WRITE_TOKEN`: Hugging Face write token (for pushing datasets)
   - `HF_READ_TOKEN`: Hugging Face read token (for accessing private datasets)

Note: The `setup()` function provided in the code is a demonstration of how you might process a custom dataset. Different datasets may require different processing steps, so you'll need to adapt this function to your specific needs.

## Customizing the LLM

Dialektik is designed to be model-agnostic. The default model is "mistralai/Mistral-Nemo-Instruct-2407", but you can easily change this by passing a different `api_model` parameter to the `synthesize()` function. 

## Output

The `synthesize()` function generates three outputs:

1. Thesis: An article exploring the main themes and insights from the selected sources.
2. Antithesis: A text presenting alternative perspectives and counterarguments to the thesis.
3. Synthesis: A reconciliation of the thesis and antithesis, presenting a new, unified viewpoint.

All outputs are saved in the 'syntheses' folder with timestamps for easy reference.

## License

This project is licensed under the [MIT License](LICENSE).

## Citation

<a href="https://zenodo.org/doi/10.5281/zenodo.11403221"><img src="https://zenodo.org/badge/806709541.svg" alt="DOI"></a>

## Contributing

Contributions to Dialektik are always welcome! Please feel free to submit a Pull Request.
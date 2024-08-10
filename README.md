# Dialektik

Merge. Synthesize. Create. Dialektik generates new content by fusing ideas from diverse sources, revealing unexpected insights and perspectives through a dialectical process.

## Features

- Loads and processes datasets from multiple sources
- Summarizes text into concise bullet points
- Performs semantic search for topic-based content selection
- Generates thesis, antithesis, and synthesis from summarized content
- Uses phi-3-vision-mlx for text generation and embeddings

## Requirements

- Python 3.8+
- `datasets`
- `huggingface_hub`
- `phi_3_vision_mlx`
- `mlx`
- `fire`

## Installation

To install Dialektik with all required dependencies:

```
pip install dialektik
```

Note: Make sure you have the necessary system requirements to run phi-3-vision-mlx and mlx.

## Usage

### Command Line Interface

Dialektik can be used from the command line after installation. Here are some example usages:

1. Generate a synthesis with default settings:

   ```
   python dialektik.py
   ```

2. Specify a topic for semantic search:

   ```
   python dialektik.py --topic "AI agents"
   ```

3. Specify sources and exclude certain terms:

   ```
   python dialektik.py --list_source arxiv --list_exclude MIRI "Machine Intelligence Research Institute"
   ```

4. Set the number of books and bullet points per book:

   ```
   python dialektik.py --num_book 5 --per_book 4
   ```

5. Use a different language model via API:

   ```
   python dialektik.py --llm_model "mistralai/Mistral-Nemo-Instruct-2407"
   ```

6. For a full list of options, use:

   ```
   python dialektik.py --help
   ```

### Python API

You can also use Dialektik in your Python scripts:

```python
from dialektik import synthesize

# Generate a synthesis with default settings
thesis, antithesis, synthesis = synthesize()

# Customize the synthesis process
output = synthesize(
   topic="AI agents",
   list_source=['arxiv'],
   list_exclude=['MIRI', 'Machine Intelligence Research Institute'],
   num_book=3,
   per_book=3
)
```

### Accessing the Dataset

The default dataset at 'JosefAlbers/StampyAI-alignment-research-dataset' is publicly available. You don't need to set up any environment variables to use `dialektik` with this dataset.

### Using Custom Datasets

If you want to use your own dataset:

1. Prepare your dataset according to the required format.
2. Modify the `PATH_DS` variable in the code to point to your dataset.
3. If your dataset is private or requires authentication, set up the following environment variables:
   - `HF_WRITE_TOKEN`: Hugging Face write token (for pushing datasets)
   - `HF_READ_TOKEN`: Hugging Face read token (for accessing private datasets)

## Customizing the Model

Dialektik now uses phi-3-vision-mlx for text generation and embeddings. This model is not easily swappable, but you can modify the `pv.load()` and `pv.generate()` calls in the code if you need to use a different model.

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
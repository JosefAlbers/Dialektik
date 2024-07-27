from pathlib import Path
from datasets import load_dataset, concatenate_datasets
import random
import json
import os
from datetime import datetime
from huggingface_hub import InferenceClient
import argparse

PATH_DS = 'JosefAlbers/StampyAI-alignment-research-dataset'

def setup(instruction="\n<|end|>\n<|user|>\nTLDR: Summarize the following text into concise, stand-alone bullet points (max 3-5 bullet points). Each bullet point should be self-contained and provide a clear and complete idea without referencing other bullet points or the original text.", list_source=['agentmodels', 'distill', 'arbital', 'blogs', 'lesswrong', 'youtube', 'arxiv', 'special_docs'], quantize_model=False, batch_size=4, path_ds=PATH_DS):
    # agentmodels 8(21) distill 16(50) arbital 358(736) blogs 914(1374) lesswrong 1785 youtube 226(558) arxiv 1696 special_docs 150(733)
    import phi_3_vision_mlx as pv
    model, processor = pv.load(blind_model=True, quantize_model=quantize_model, quantize_cache=False, use_adapter=False)
    def aggregate(example):
        str_md = f"# {example['title']}\n\n{example['text']}"
        example['str_md'] = str_md
        example['len_md'] = processor(str_md)['input_ids'].size
        return example
    def summarize(example):
        markdowns = example['str_md']
        prompts = [f'{m}{instruction}' for m in markdowns]
        summaries = pv.generate(prompts, preload=(model, processor), stream=False, verbose=False, max_tokens=512)
        example['sum_md'] = summaries
        return example
    list_ds = []
    try:
        _ds_prev = load_dataset(path_ds, token=os.getenv("HF_WRITE_TOKEN"), split='train')
        list_source = [i for i in list_source if i not in _ds_prev['source']]
        list_ds.append(_ds_prev)
    except:
        print('Dataset not found.')
    for src in list_source:
        ds = load_dataset('StampyAI/alignment-research-dataset', src, trust_remote_code=True, split='train')
        ds = ds.select_columns(['id', 'source', 'title', 'text', 'url', 'date_published', 'authors', 'summary', 'source_type'])
        ds = ds.map(aggregate)
        ds = ds.filter(lambda example: 600 < example["len_md"] < 6000)
        if batch_size > 1:
            ds = ds.sort('len_md')
        ds = ds.map(summarize, batched=True, batch_size=batch_size)
        ds = ds.filter(lambda example: ('<unk>' not in example['sum_md']) and ('<|end|>' in example['sum_md']))
        list_ds.append(ds)
    ds = concatenate_datasets(list_ds)
    ds.push_to_hub(path_ds, token=os.getenv("HF_WRITE_TOKEN"), private=True)

def load_books(list_source=None, list_exclude=None, path_ds=PATH_DS):
    ds = load_dataset(path_ds, token=os.getenv("HF_READ_TOKEN", None), split='train')
    if list_source:
        list_source = [list_source] if isinstance(list_source, str) else list_source
        ds = ds.filter(lambda example: example['source'] in list_source)
    if list_exclude:
        list_exclude = [list_exclude] if isinstance(list_exclude, str) else list_exclude
        ds = ds.filter(lambda example: not any(word in example['sum_md'] for word in list_exclude))
    print(f"Loaded {len(ds)} from {', '.join(set(ds['source']))}")
    books = ds['sum_md']
    books = [i.split('\n- ') for i in books]
    clean_str = lambda s: s[2:] if s.startswith('- ') else s[:-7] if s.endswith('<|end|>') else s
    books = [[clean_str(s).strip() for s in book] for book in books]
    return books

def pick_books(list_source=None, list_exclude=['MIRI', 'Machine Intelligence Research Institute'], list_idx=None, per_book=3):
    books = load_books(list_source, list_exclude)
    list_idx = list_idx if list_idx else random.sample(range(len(books)), 3)
    print(f"Picked {list_idx}")
    picks = [books[i] for i in list_idx]
    synth = ''
    for pick in picks:
        pick=pick[:per_book]
        synth += '- ' + '\n    - '.join(pick) + '\n'
    synth = synth.strip()
    print(f'Synth:\n{synth}')
    return synth, list_idx

def mistral_api(prompt, history, verbose=True, api_model="mistralai/Mistral-Nemo-Instruct-2407"):
    # "mistralai/Mistral-Nemo-Instruct-2407" "mistralai/Mistral-7B-Instruct-v0.3"
    history = '<s>' if history is None else history
    history += f"[INST] {prompt} [/INST]"
    client = InferenceClient(api_model, token = os.environ.get('HF_READ_TOKEN', False))
    generate_kwargs = dict(
        temperature=0.9,
        max_new_tokens=1024,
        top_p=0.95,
        repetition_penalty=1.0,
        do_sample=True,
        seed=42,
        stream=False,
        details=False,
        # details=True,
        return_full_text=False,
    )
    result = client.text_generation(history, **generate_kwargs)
    result = result.strip()
    # result = result.generated_text.strip() # if details=True
    history += f" {result}</s> "
    if verbose:
        print(f'### Prompt ###\n{prompt}\n### Output ###\n{result}')
    return {'responses':result, 'history':history}

def save_output(output, base_folder = 'syntheses'):
    os.makedirs(base_folder, exist_ok=True)
    date_str = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    filename = os.path.join(base_folder, f'{date_str}.md')
    with open(filename, 'w') as f:
        f.write(output)

def synthesize(list_source=None, list_exclude=['MIRI', 'Machine Intelligence Research Institute'], list_idx=None, per_book=3, api_model="mistralai/Mistral-Nemo-Instruct-2407"):
    draft, list_idx = pick_books(list_source, list_exclude, list_idx, per_book)
    prompt = f"{draft}\n\nBased on the above bullet points, create a detailed and engaging article that explores the main themes and insights. For each bullet point, provide context, elaborate on the key ideas, and discuss their implications. Ensure the article flows logically, connects related concepts, and presents a coherent narrative."
    output = mistral_api(prompt, None, False, api_model=api_model)['responses'].strip()
    print(f'Result:\n{output}')
    save_output(f'{output}\n---\n{draft}\n---\n* {list_source=} {list_exclude=} {list_idx=} {per_book=} {api_model=} *\n---')
    return output


def main():
    parser = argparse.ArgumentParser(description="Dialektik: A tool for summarizing and synthesizing content.")
    parser.add_argument("--source", nargs='+', help="List of sources to use")
    parser.add_argument("--exclude", nargs='+', default=['MIRI', 'Machine Intelligence Research Institute'], help="List of terms to exclude")
    parser.add_argument("--per-book", type=int, default=3, help="Number of bullet points per book")
    parser.add_argument("--model", default="mistralai/Mistral-Nemo-Instruct-2407", help="API model to use")
    parser.add_argument("--setup", action="store_true", help="Run setup function")
    args = parser.parse_args()

    if args.setup:
        setup()
    else:
        output = synthesize(
            list_source=args.source,
            list_exclude=args.exclude,
            per_book=args.per_book,
            api_model=args.model
        )

if __name__ == "__main__":
    main()

import bisect
from dataclasses import dataclass
import pandas as pd
import os
import random
import re
import streamlit as st
from transformers import AutoTokenizer
import torch
from training_interpretability.model import BasicTransformer, Config
from torchtyping import TensorType as TT
from typing import List, Optional, Tuple, Union

# Plotting related imports
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


@dataclass
class AppConfig:
    checkpoints_dir: str

app_config = AppConfig(
    checkpoints_dir="experiments/induction-heads"
)

model_config = Config(
    d_model=(d_model := 768 // 2),
    debug=False,
    layer_norm_epsilon=1e-5,
    d_vocab=50257,
    init_range=0.02,
    n_context=512,
    d_head=64,
    d_mlp=d_model * 2,
    n_heads=d_model // 64,
    n_layers=3,
    device="cpu",
    _record_attn_pattern=True,
)

common_prompts = [
    "John and Mary went to the store. John and Mary went to the store.",
    "foo bar baz qux foo bar baz qux",
    "apple banana orange pear apple banana orange pear",
]

@st.cache_resource
def model_checkpoints() -> List[int]:
    checkpoints = list(sorted([
        int(id) for file in os.listdir(app_config.checkpoints_dir)
        if file.endswith(".pt") and re.match("[0-9]+", id := file.split("_")[-1].split(".")[0])
    ]))
    return checkpoints

@st.cache_resource
def load_model(checkpoint: int=None) -> BasicTransformer:
    if not checkpoint or checkpoint < 0:
        checkpoint = "model_params.pt"
    else:
        checkpoint = f"model_params_{checkpoint}.pt"

    model_state = torch.load(os.path.join(app_config.checkpoints_dir, checkpoint), map_location=torch.device('cpu'))
    model = BasicTransformer(model_config)
    model.load_state_dict(model_state)
    return model

@st.cache_resource
def load_tokenizer() -> AutoTokenizer:
    return AutoTokenizer.from_pretrained("gpt2")

def generate_text(checkpoint: int, text: str, forward_tokens: int=10) -> pd.DataFrame:
    """
    Takes in a model checkpoint and text to predict, and generates next token predictions.
    
    Returns a dataframe comparing tokens to next token.
    """
    tokenizer = load_tokenizer()
    
    for i in range(forward_tokens):
        if i == 0:
            raw_tokens = tokenizer.encode(text)
        else:
            raw_tokens = text
        tokens = torch.tensor(raw_tokens, device=model_config.device).type(torch.long).unsqueeze(0)
        model = load_model(checkpoint)
        output = model(tokens).argmax(dim=-1).tolist()[0]
        text = tokens + output[-1]

    return pd.DataFrame({"token": [tokenizer.decode(x) for x in raw_tokens],
                         "predicted": [tokenizer.decode(x) for x in output],
                         "correct": [x == y for x, y in zip(raw_tokens[1:] + [-1], output)]})

@st.cache_data
def induction_check(checkpoint: int, induction_threshold: float=0.5, samples: int=10) -> Tuple[bool, float]:
    model = load_model(checkpoint)
    num_matches = []
    for _ in range(samples):
        random_seq = torch.tensor([random.sample(range(model_config.d_vocab), k=20)*2]).type(torch.long)
        outputs = model(random_seq).argmax(dim=-1)
        pairs = list(zip(random_seq[:, 1:][0].tolist() + [0], outputs[0].tolist()))
        num_matches.append(sum(int(correct == predicted) for correct, predicted in pairs[random_seq.shape[1] // 2:]))
    
    induction_score = sum(num_matches) / len(num_matches)
    return induction_score >= random_seq.shape[1] // 2 * induction_threshold, induction_score / (random_seq.shape[1] // 2)

@st.cache_data
def attn_patterns(checkpoint: int, prompt: Optional[Union[torch.Tensor, str]]=None) -> TT["layer", "head", "query", "key"]:
    model = load_model(checkpoint)
    if isinstance(prompt, str):
        prompt = torch.tensor(load_tokenizer().encode(prompt), device=model_config.device).type(torch.long).unsqueeze(0)
    
    max_samples = 10 if prompt is None else 1
    for _ in range(max_samples):
        if prompt is None:
            input = torch.tensor([random.sample(range(model_config.d_vocab), k=20)*2]).type(torch.long)
        else:
            input = prompt
        outputs = model(input).argmax(dim=-1)
        pairs = list(zip(input[:, 1:][0].tolist() + [0], outputs[0].tolist()))
        matches = sum(int(correct == predicted) for correct, predicted in pairs[input.shape[1] // 2:])
        if matches > input.shape[1] // 2 * induction_threshold:
            # Model exhibits induction on this sequence
            break

    # Attention patterns have been saved in the model.
    return torch.stack([b.attn.attn_pattern.squeeze(0) for b in model.blocks], dim=0)
    
@st.cache_data
def induction_patterns(checkpoint: int, prompt: Optional[str]=None) -> pd.DataFrame:
    if prompt and len(load_tokenizer().encode(prompt)) % 2 == 1:
        raise ValueError("Prompt must have even number of tokens to check induction score.")

    # Tensor shape: [layer, head, query, key]
    patterns = attn_patterns(checkpoint, prompt)

    # Assuming the prompt is a duplication of its first half, the halfway lower
    # diagonal of the attention patterns should be non-trivial. 
    induction_scores = patterns.diagonal(offset=-patterns.shape[-1] // 2 + 1, dim1=-2, dim2=-1).mean(dim=-1)
    #induction_scores = patterns.diagonal(offset=0, dim1=-2, dim2=-1).mean(dim=-1)
    assert len(induction_scores.shape) == 2 # Tensor shape: [layer, head]

    return pd.DataFrame([
        (layer_index, block_index, float(induction_scores[layer_index, block_index]))
        for layer_index in range(patterns.shape[0])
        for block_index in range(patterns.shape[1])
    ], columns=["layer_index", "head_index", "value"])

@st.cache_resource
def plot_attention_pattern(checkpoint: int, layer: int, head: int, prompt: Optional[str]=None) -> matplotlib.figure.Figure:
    patterns = attn_patterns(checkpoint, prompt)
    pattern = patterns[layer, head]
    pattern = pd.DataFrame([
        (query_pos, key_pos, float(pattern[query_pos, key_pos]))
        for query_pos in range(pattern.shape[0])
        for key_pos in range(pattern.shape[1])
    ], columns=["query_pos", "key_pos", "value"])

    pivot_df = pattern.pivot(index="query_pos", columns="key_pos", values="value")

    # Create a figure and axes object
    fig, ax = plt.subplots(figsize=(10, 6))

    cmap = plt.get_cmap("RdBu_r")
    im = ax.imshow(pivot_df, cmap=cmap, aspect="auto", origin="lower", interpolation="none")
    im.set_clim(0, 1)

    # Customize the x and y axes
    ax.set_xlabel("Key position")
    ax.set_ylabel("Query position")

    # Force integer labels on the x-axis and y-axis
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.invert_yaxis()
    ax.set_title(f"Attention pattern scores for layer {layer + 1} and head {head + 1}")

    # Add a colorbar to show the mapping from values to colors
    cbar = fig.colorbar(im)
    cbar.set_label("Attention pattern score")

    return fig

@st.cache_resource
def plot_induction_patterns(attn_patterns: pd.DataFrame) -> matplotlib.figure.Figure:
    pivot_df = attn_patterns.pivot(index="layer_index", columns="head_index", values="value")

    # Create a figure and axes object
    fig, ax = plt.subplots(figsize=(10, 6))

    cmap = plt.get_cmap("RdBu_r")
    im = ax.imshow(pivot_df, cmap=cmap, aspect="auto", origin="lower", interpolation="none")
    im.set_clim(0, 1)

    # Customize the x and y axes
    ax.set_xlabel("Attention Head Number")
    ax.set_ylabel("Layer Number")

    # Force integer labels on the x-axis and y-axis
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_title(f"Induction head scores")

    # Add a colorbar to show the mapping from values to colors
    cbar = fig.colorbar(im)
    cbar.set_label("Value")

    return fig

if __name__ == "__main__":
    st.set_page_config(layout="wide")

    with st.sidebar:
        checkpoint = st.slider("Model checkpoint", min_value=0, max_value=max(model_checkpoints()), step=1000, value=max(model_checkpoints()))
        checkpoint = model_checkpoints()[bisect.bisect_left(model_checkpoints(), checkpoint)]

        induction_threshold = st.slider("Induction threshold (global)", min_value=0.0, max_value=1.0, step=0.05, value=0.5)
        #induction_threshold_head = st.slider("Induction threshold (head)", min_value=0.0, max_value=1.0, step=0.05, value=0.5)

        layer = st.slider("Layer", min_value=1, max_value=model_config.n_layers, step=1, value=1)
        head = st.slider("Head", min_value=1, max_value=model_config.n_heads, step=1, value=1)

        induction_prompt = st.radio("Show induction score on", ["Random sequence", "Input prompt"], index=0)
        show_induction_score_means = st.checkbox("Show induction score means")

    cols = st.columns([0.3, 0.7])

    with cols[0]:
        common_prompt = st.radio("Template prompt", ["Off"] + common_prompts)
        prompt = st.text_area("Input prompt", common_prompts[0])
        active_prompt = common_prompt if common_prompt != "Off" else prompt
        model = load_model(checkpoint)
        st.dataframe(generate_text(checkpoint, active_prompt, forward_tokens=1), height=600)

    with cols[1]:
        st.subheader(f"Exploring induction heads (checkpoint: {checkpoint})")

        induction_prompt = None if induction_prompt == "Random sequence" else active_prompt

        induction_detected, induction_score = induction_check(checkpoint, induction_threshold=induction_threshold)
        color = "green" if induction_detected else "red"
        st.write(f"Induction heads detected: :{color}[{'YES' if induction_detected else 'NO'}] (score: {induction_score:.2f} {'>=' if induction_detected else '<'} {induction_threshold:.2f})")
        st.write(f"Induction prompt: {induction_prompt}")

        st.pyplot(plot_attention_pattern(checkpoint, layer-1, head-1, induction_prompt))

        df = induction_patterns(checkpoint, induction_prompt)
        if show_induction_score_means:
            st.write(df)

        st.pyplot(plot_induction_patterns(df))

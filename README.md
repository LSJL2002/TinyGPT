# TinyGPT

A small educational implementation of a GPT-style character-level language model using PyTorch.

## What this project does

`tinygpt.ipynb` defines a minimal transformer-based model that learns to predict the next character in a text sequence.

Key components:
- Data loading from `input.txt` (downloads Tiny Shakespeare text if the file is missing)
- Character vocabulary creation and tokenization
- A masked self-attention head and multi-head attention layer
- Transformer blocks with layer normalization, residual connections, and feed-forward networks
- A training loop using cross-entropy loss and AdamW optimizer
- A sampling function to generate new text from a prompt

## How the code works

### `NextTokenDataset`
This dataset class prepares training examples for next-character prediction. For each index, it returns a context sequence `x` of length `block_size` and the target sequence `y` containing the next character for every position in `x`.

### `Head`
A single attention head computes masked self-attention over the input sequence. It projects input embeddings into query, key, and value vectors, applies a causal mask so tokens cannot attend to future positions, and aggregates values with attention weights.

### `MultiHeadAttention`
This module runs several `Head` instances in parallel and concatenates their outputs. It then projects the combined result back to the model embedding dimension, allowing the model to attend to different representation subspaces simultaneously.

### `FeedForward`
A small position-wise feed-forward network that applies two linear layers with a ReLU activation in between. It expands the embedding dimension and then projects back, giving the model extra capacity to transform token representations.

### `Block`
A transformer block that combines multi-head self-attention and feed-forward computation. It applies layer normalization before each sub-layer and uses residual connections so the model can learn from both the original and transformed representations.

### `TinyGPT`
The full GPT-style model. It embeds tokens and positions, applies a stack of transformer blocks, normalizes the final output, and projects to vocabulary logits. The output logits are used to compute cross-entropy loss during training and to sample the next token during text generation.

## How to run `tinygpt.ipynb`

1. Open the notebook in your Jupyter environment.
2. Ensure a Python environment with PyTorch installed is active.
3. Run the notebook cells in order.

### Notes

- The notebook will download `input.txt` automatically if it is not present.
- The model trains on the Tiny Shakespeare dataset using 64-character context windows.
- By default, it uses `cuda` when available, otherwise it falls back to CPU.

## Expected behavior

When the notebook finishes training, it prints generated text seeded with the prompt `"ROMEO:\n"`.

## Requirements

- Python 3.8+
- PyTorch

If you want, you can also run the notebook with a standard Jupyter server or VS Code Jupyter support.

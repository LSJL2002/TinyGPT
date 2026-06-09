# TinyGPT: Detailed Explanation

This file explains how the TinyGPT code works in a way that a first-time student can understand. It goes through each class step by step, describes what it does, and explains how the model produces output.

---

## 1. What this project does

TinyGPT is a small version of a transformer-based language model. It works at the character level, which means it reads and predicts one character at a time instead of whole words.

The goal is to train a model that can guess the next character in a text sequence. Once trained, the model can generate new text by predicting one character after another.

---

## 2. Data preparation

Before the model can learn, the text is converted into numbers.

- The code reads `input.txt` and collects all unique characters.
- Each character is given a number using a dictionary called `stoi` (string to index).
- Another dictionary `itos` (index to string) is used to convert numbers back into characters.
- The raw text becomes a tensor of integer token IDs.

This is important because neural networks only work with numbers, not letters.

---

## 3. `NextTokenDataset`

This class creates the training examples the model learns from.

What it stores:
- `data`: the full sequence of character token IDs
- `block_size`: how many characters the model looks at at once

What it returns:
- `x`: a sequence of `block_size` tokens
- `y`: the same sequence shifted one character to the right

So the model learns to predict the next character at each position in the block.

For example: If the block size was 3 and the full sequence was the word "Blocks"
x: "B" "l" "o"
y: "l" "o" "c"

Since y is shifted by exactly one position, it creates three individual training pairs at the same time. 

The first one being that the model sees "B" and it learns to predict "l"

The second one being that the model sees "Bl" and it learns to predict "o"

The third one being that the model sees "Blo" and it learns to predict "c"

---

## 4. `Head`

This class implements one attention head.

### What attention does
Attention allows the model to look at every character in the input block and decide which ones are important when predicting the next character.

### Inputs and outputs
- Input `x` has shape `(B, T, C)`:
  - `B`: batch size (number of sequences processed at once)
  - `T`: sequence length (`block_size`)
  - `C`: embedding dimension (size of each character vector)

### Steps
- Step 1 Creating the Queries, Keys and values
   - Key(k), Query(q) and Value(v) are created in the class. 
   - Key represents what a token offers
   - Query represents what a tokens actively searching for
   - Value represents actual message. 
   - The code in the ipynb passes the input x through the three independent, linear layers. Every token in the sequence gets its own private q, k, and v vectors of size head_size
- Step 2 Computing the attention raw scores (Matrix Multiplication)
   - At this specific step of the process the code does not care about the actual meaning of the content of the tokens yet.
   - It's main purpose its to calculate the routing map or a matrix of importance percentages.
   - In this step 2 linear layers of q and k are multiplied. (Q x K^t)
   - If the Query and a Key match well, they get a high score if they dont they get a low score.
   - IE (Estimated Numbers)

     ```text
            "B"   "l"   "o"
     "B"  [1.2] [0.4] [-2.1]
     "l"  [0.8] [2.5] [0.1]
     "o"  [-0.5] [4.2] [1.8]
     ```

   - Scores are claculated based on the dot product. If two vectors point in a similar direction in space, multiplying them will result in a large positive number
   - If they are facing in opposite direction it will yield a negative value
   - If they are unrelated(perpendicular) it yields a value close to zero.
- Step 3 Casual Masking
   -  wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf")). This line of code fetches a lower-triangular matrix.
   - Lower-triangular Matrix

     ```text
     [1 0 0]
     [1 1 0]
     [1 1 1]
     ```

   - The upper half, which is 0, forces the score to become negative infinity(-inf). This is done in order to represent impossible relationship. 
   - This is done to enforce the model to constraint itself from cheating by looking in the future, and actually train and learn how to predict instead of looking at the answer.
   - For example, using the same word "Blocks", if the model starts at B, the model blinds the letter "l" and "o", and only has acess to the letter "B". If it predicts "Bl" then it can see the letters "Bl" but not "o"
   - What it would look like when masked

     ```text
            "B"   "l"   "o"
     "B"  [1.2] [-inf] [-inf]
     "l"  [0.8] [2.5] [-inf]
     "o"  [-0.5] [4.2] [1.8]
     ```

- Step 4 Turning scores into percentages
   - The softmax function takes the masked scores and squashes them into a value between 0% ~ 100%. 
   - For the case of "Blo" it will show that the character "o" has a strong connection to the letter "l". 
- Step 5 Pulling the final meanings
   - For the final step the code multiplies the percentage weights wei by the actual value vectors (v). 
   - In the case for the example the output would be a final mixed vector for "o" that contains a very high percentage of the meaning "l", which helps it to predict the next letter, which in this case should be "c".

## 5. `MultiHeadAttention`

This class combines several `Head` instances.

### Why multiple heads?
Each head can learn a different way of paying attention.
- One head might learn to focus on punctuation.
- Another might learn to remember earlier letters.

### What it does
-  Step 1 Splitting the Workload
   - First the code will define how the total embedding dimension(embedding dimensions) is divided among the heads.
   - This is done in order to avoid having one giant attention mechanism looking at everything.
   - For example if the total size of emb_dim is 256 and there are 4 heads, each head will focus on a smaller feature space (head_size) of 64.
-  Step 2 Parallel Attention Processing (forward)
   - When data passes through the network, the code/model will run all of the heads at the exact same time over the input sequence x. When each head calculates its own q,k,v it performs the scaled dot-product attention, and outputs its own focus map.
-  Step 3 Gluing the results together (torch.cat)
   -  Once all heads finish calculating, the code will bring in all of their results together through torch.cat.
- Step 4 Final Projection Layer (self.proj)
   -  Since multiple heads are glued together it is necessary to format it in a way that it can be passed onto the next step. By using linear projection, the code is able to get the combined matrix of all heads, mixes their insights together mathematically, and maps them back into a clean strcture for the next step.
---

## 6. `FeedForward`

This class is a small neural network that processes each position independently.

### Structure
- A linear layer expands the embedding from size `emb_dim` to `4 * emb_dim`.
- A ReLU activation introduces non-linearity.
- Another linear layer reduces the size back to `emb_dim`.
- Dropout is applied at the end.

### Why it is needed
Attention tells the model what to look at. Feed-forward layers help the model learn more complex transformations of the data at each position.

---

## 7. `Block`

This is the basic building block of the transformer.

It does two things in sequence:
1. Multi-head self-attention
2. Feed-forward processing

### Detailed structure
- `ln1` is a layer normalization applied before attention.
- `sa` is the multi-head attention module.
- `ln2` is another layer normalization before the feed-forward network.
- `ffwd` is the feed-forward module.

### Residual connections
The block adds the input back to the output of each sub-layer:
- `x = x + self.sa(self.ln1(x))`
- `x = x + self.ffwd(self.ln2(x))`

This helps training by allowing the model to keep the original signal and only learn small changes.

---

## 8. `TinyGPT`

This is the full model that turns token IDs into predictions.

### Components
- `token_embedding`: turns token IDs into vectors
- `position_embedding`: gives each position in the sequence a unique vector
- `blocks`: a stack of transformer blocks
- `ln_f`: final layer normalization
- `lm_head`: linear layer that converts embeddings into vocabulary logits

### How it works
1. Convert token IDs into vectors using `token_embedding`.
2. Create position vectors for each token position in the block.
3. Add token and position embeddings together.
4. Pass the result through the stack of transformer blocks.
5. Normalize the final result.
6. Use `lm_head` to produce logits for every character in the vocabulary.

### Output shape
The final output is a tensor of shape `(B, T, vocab_size)`.
- For every batch example and every token position,
- the model produces one score for each possible character.

These scores are called logits.

---

## 9. How the model produces output

### From logits to probabilities
Logits are raw scores. To turn them into probabilities, the code uses `softmax`.

- Softmax converts the scores into a probability distribution.
- Each probability is between 0 and 1.
- The probabilities for a given position sum to 1.

### Training output
During training, the model predicts the next character for each position in the input block.
- If the input is `x = [a, b, c]`, the target is `y = [b, c, d]`.
- The model outputs logits for each position.
- `sequence_cross_entropy` compares the logits to the true targets.
- The loss tells the model how wrong it was.
- The optimizer updates the weights to make the loss smaller.

### Sampling output
For generation, the model is run step-by-step:
1. A start text is turned into token IDs.
2. The model predicts the next token logits for the whole context.
3. The code takes only the last position logits.
4. Softmax turns them into probabilities.
5. `torch.multinomial` samples one token from the distribution.
6. The chosen token is added to the output.
7. The context shifts and the process repeats.

This is how TinyGPT creates new text character by character.

---

## 10. Simple diagram of the model flow

```text
input tokens  -> [Token Embedding] --------+
                                         |
positions      -> [Position Embedding] --->+--> [Add embeddings] -> [Transformer blocks] -> [LayerNorm] -> [lm_head] -> logits

Transformer block:
  input -> LayerNorm -> MultiHeadAttention -> + -> LayerNorm -> FeedForward -> + -> output

MultiHeadAttention:
  input -> Head 1 ->
               Head 2 ->
               ...     -> concatenate -> linear projection -> output

Head:
  input -> query/key/value projections -> attention scores -> mask -> softmax -> weighted sum -> output
```

---

## 11. Why this is a good learning model

- It is small enough to understand by reading the code.
- It uses the same ideas as large GPT models, but on characters instead of words.
- It shows how attention, embeddings, and residual connections work together.
- It also shows the full training and generation loop.

If you want, I can also add a second diagram showing how training differs from sampling step by step.
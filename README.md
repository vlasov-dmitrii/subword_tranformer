# Transformer Language Models and Tokenizers

This repository contains multiple projects exploring transformer-based language models and tokenizers:
**Character-level Transformer Language Model**

- A Transformer implemented from scratch in PyTorch.
- Trained on raw text using character-level embeddings. 
- Capable of generating text continuations.

**Custom Tokenizers**

- Implementations of Byte-Pair Encoding (BPE), regex-based tokenization, and a GPT-4–style tokenizer.
- Designed to show how modern tokenization methods work under the hood.
  
**Transformer Language Model with SentencePiece (Wikitext-103):**

- A full Transformer LM trained on the Wikitext-103 dataset.
- Uses SentencePiece BPE for subword tokenization.
- Includes advanced training features like learning rate scheduling, gradient clipping, and nucleus sampling for generation.

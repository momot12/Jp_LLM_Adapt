import torch
import transformers
import spacy

print("Imports successful!")

# Test 1: Can we access the GPU/CPU device?
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Test 2: Can we load a basic Spacy model?
try:
    nlp = spacy.load("ja_core_news_sm")
    print("Spacy model loaded successfully.")
except Exception as e:
    print(f"Spacy error: {e}")

# Test 3: Can we create a simple tensor (checks CUDA/BLAS kernels)?
x = torch.tensor([1.0, 2.0], device=device)
print(f"Tensor created on {x.device}: {x}")
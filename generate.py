import os
import torch

from model.tokenizer import CharTokenizer
from model.transformer import TinyGPT

PROJECT_ROOT = os.path.dirname(__file__)
MODEL_DIR = os.path.join(PROJECT_ROOT, "model")
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "data.txt")
MODEL_PATH = os.path.join(MODEL_DIR, "tinygpt.pt")


def load_tokenizer():
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        text = f.read()
    if len(text) == 0:
        raise ValueError("data/data.txt is empty.")
    return CharTokenizer(text)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load tokenizer
    tokenizer = load_tokenizer()

    # Load model config + weights
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    vocab_size = checkpoint["vocab_size"]
    block_size = checkpoint["block_size"]
    embed_dim = checkpoint["embed_dim"]
    num_layers = checkpoint["num_layers"]

    model = TinyGPT(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        block_size=block_size,
        num_layers=num_layers,
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # ----- Generation -----
    # Start with a single token; take first character of corpus as seed
    seed_char = " "
    if seed_char not in tokenizer.stoi:
        seed_char = list(tokenizer.stoi.keys())[0]

    start_id = torch.tensor([[tokenizer.stoi[seed_char]]], dtype=torch.long, device=device)
    out = model.generate(start_id, max_new_tokens=200)[0].tolist()
    text = tokenizer.decode(out)

    print("=== GENERATED TEXT ===")
    print(text)


if __name__ == "__main__":
    main()

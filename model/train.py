import os
import sys
import torch
import torch.nn.functional as F

# Make sure this file can import tokenizer and transformer
THIS_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.dirname(THIS_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from model.tokenizer import CharTokenizer
from model.transformer import TinyGPT


def main():
    # ----- Hyperparameters -----
    block_size = 8     # max context length
    batch_size = 16
    embed_dim = 64
    num_layers = 2
    learning_rate = 1e-3
    max_iters = 1000     # increase if you want better fitting
    eval_interval = 100

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # ----- Load data -----
    data_path = os.path.join(PROJECT_ROOT, "data", "data.txt")
    with open(data_path, "r", encoding="utf-8") as f:
        text = f.read()

    if len(text) == 0:
        raise ValueError("data/data.txt is empty. Put some text in it.")

    tokenizer = CharTokenizer(text)
    print(f"Vocab size: {tokenizer.vocab_size}")

    # Encode entire corpus as a single long tensor
    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)

    # Train/val split (simple)
    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]

    def get_batch(split: str):
        src = train_data if split == "train" else val_data
        # pick random starting indices
        ix = torch.randint(0, len(src) - block_size - 1, (batch_size,))
        x = torch.stack([src[i:i + block_size] for i in ix])
        y = torch.stack([src[i + 1:i + block_size + 1] for i in ix])
        return x.to(device), y.to(device)

    # ----- Model -----
    model = TinyGPT(
        vocab_size=tokenizer.vocab_size,
        embed_dim=embed_dim,
        block_size=block_size,
        num_layers=num_layers,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    # ----- Training loop -----
    for step in range(max_iters):
        model.train()
        x, y = get_batch("train")

        logits = model(x)  # (B, T, V)
        B, T, V = logits.shape
        loss = F.cross_entropy(logits.view(B * T, V), y.view(B * T))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % eval_interval == 0 or step == max_iters - 1:
            model.eval()
            with torch.no_grad():
                x_val, y_val = get_batch("val")
                logits_val = model(x_val)
                Bv, Tv, Vv = logits_val.shape
                val_loss = F.cross_entropy(
                    logits_val.view(Bv * Tv, Vv),
                    y_val.view(Bv * Tv)
                )
            print(f"step {step} | train loss {loss.item():.4f} | val loss {val_loss.item():.4f}")

    # ----- Save model -----
    model_path = os.path.join(THIS_DIR, "tinygpt.pt")
    torch.save({
        "model_state_dict": model.state_dict(),
        "vocab_size": tokenizer.vocab_size,
        "block_size": block_size,
        "embed_dim": embed_dim,
        "num_layers": num_layers,
    }, model_path)

    print(f"Saved trained model to {model_path}")


if __name__ == "__main__":
    main()

# cpo_trainer.py
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse

MODEL_NAME = "mistralai/Mistral-7B-v0.1"  # Public, no login

def cpo_loss(logps_good, logps_bad, beta=0.1):
    """
    CPO loss (Direct Preference Optimization variant)
    logps_good: log-prob of preferred translation
    logps_bad : log-prob of dispreferred translation
    """
    diff = logps_good - logps_bad
    return -torch.log(torch.sigmoid(beta * diff)).mean()

def tokenize_batch(tokenizer, batch):
    src = [f"Translate to French: {s}" for s in batch["src"]]
    return tokenizer(src, return_tensors="pt", padding=True, truncation=True, max_length=256)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--epochs", type=int, default=1)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto")
    model.eval() if args.demo else model.train()

    # Load dataset
    from datasets import load_from_disk
    dataset = load_from_disk("cpo_triplets")
    loader = DataLoader(dataset, batch_size=4, shuffle=True)

    if args.demo:
        print("=== DEMO ===")
        batch = next(iter(loader))
        inputs = tokenize_batch(tokenizer, batch)
        with torch.no_grad():
            outputs = model(**inputs, labels=inputs["input_ids"])
            logps = outputs.logits.log_softmax(-1)
            print("Log-probs computed – ready for CPO loss")
        return

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-6)
    for epoch in range(args.epochs):
        total_loss = 0.0
        for batch in tqdm(loader, desc=f"Epoch {epoch+1}/{args.epochs}"):
            # Tokenize source + good/bad
            src_inputs = tokenize_batch(tokenizer, batch)
            good_inputs = tokenizer(batch["good"], return_tensors="pt", padding=True, truncation=True, max_length=256)
            bad_inputs  = tokenizer(batch["bad"],  return_tensors="pt", padding=True, truncation=True, max_length=256)

            # Forward passes
            with torch.no_grad():
                # Freeze reference model for DPO-style contrast
                ref_outputs = model(**src_inputs)
            # Trainable policy
            policy_good = model(**src_inputs, labels=good_inputs["input_ids"])
            policy_bad  = model(**src_inputs, labels=bad_inputs["input_ids"])

            # Approximate log-probs (sum over chosen tokens)
            logps_good = policy_good.logits.log_softmax(-1).gather(2, good_inputs["input_ids"].unsqueeze(-1)).squeeze(-1).sum(-1)
            logps_bad  = policy_bad.logits.log_softmax(-1).gather(2, bad_inputs["input_ids"].unsqueeze(-1)).squeeze(-1).sum(-1)

            loss = cpo_loss(logps_good, logps_bad)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item()
        print(f"Epoch {epoch+1} loss: {total_loss/len(loader):.4f}")

if __name__ == "__main__":
    main()
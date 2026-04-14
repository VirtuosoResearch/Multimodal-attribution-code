import torch
import torch.nn as nn
from transformers import (
    AutoModel,
    AutoTokenizer,
    AutoModelForCausalLM,
    LlamaForCausalLM,
    LlamaTokenizer,
)
class TextModel(torch.nn.Module):
    def __init__(self, model_name):
        super(TextModel, self).__init__()
        self.model_name = model_name
        name = self.model_name.lower()
        
        # --- Encoder-only (BERT/RoBERTa/etc) ---
        if "bert" in name or "roberta" in name or "deberta" in name:
            self.model_type = "encoder"
            self.model = AutoModel.from_pretrained(self.model_name)
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        # --- LLaMA-family decoders ---
        elif "llama" in name:
            self.model_type = "decoder"
            self.model = LlamaForCausalLM.from_pretrained(self.model_name)
            self.tokenizer = LlamaTokenizer.from_pretrained(self.model_name, output_hidden_states=True)
            self.tokenizer.pad_token = self.tokenizer.eos_token # LLaMA has no pad token by default
        # --- Mistral-family decoders (via remote code) ---
        elif "mistral" in name:
            self.model_type = "decoder"
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name, use_auth_token=True, trust_remote_code=True, local_files_only=False, torch_dtype=torch.float16)
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # --- Other causal-LM decoders (GPT-2, OPT, Bloom, etc) ---
        elif any(tok in name for tok in ["gpt2", "opt", "bloom"]):
            self.model_type = "decoder"
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name, output_hidden_states=True)
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.tokenizer.pad_token = self.tokenizer.eos_token
        else:
            raise ValueError(f"Unsupported model type: {self.model_name!r}")

    def forward(self, x, return_tokens: bool = False):
        if self.model_type == 'encoder':
            outputs = self.model(**x)
            last_hidden = outputs.last_hidden_state  # (B, T, D)
            if return_tokens:
                return last_hidden
            return last_hidden[:, 0, :] # CLS token
        elif self.model_type == 'decoder':
            # causal decoders: mean-pool over non-pad tokens (or return tokens)
            out = self.model(**x, output_hidden_states=True)
            last_hidden = out.hidden_states[-1]                         # last hidden state (B, T, D)
            attention_mask = x['attention_mask'].unsqueeze(-1)          # (B, T, 1)
            if return_tokens:
                return last_hidden * attention_mask                     # zero out pad tokens
            masked_hidden = last_hidden * attention_mask                # zero out pad tokens
            sum_embeddings = masked_hidden.sum(dim=1)                   # (B, D)
            sum_mask = attention_mask.sum(dim=1)                        # (B, 1)
            return sum_embeddings / sum_mask                            # (B, D)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

if __name__ == "__main__":
    import torch
    model_map = {
        "bloom0.56b":     "bigscience/bloom-560m",                       # 1024 embedding dimension
        "bloom1.1b":      "bigscience/bloom-1b1",                        # 1536 embedding dimension
        "bloom1.7b":      "bigscience/bloom-1b7",                        # 2048 embedding dimension
        "bloom3b":        "bigscience/bloom-3b",                         # 2560 embedding dimension
        "openllama3b":    "openlm-research/open_llama_3b_v2",            # 3200 embedding dimension
        "openllama7b":    "openlm-research/open_llama_7b",               # 4096 embedding dimension 
        "openllama13b":   "openlm-research/open_llama_13b",
        "mistral7b":     "mistralai/Mistral-7B-v0.1",
    }
    device = "cuda" if torch.cuda.is_available() else "cpu"
    for alias, repo_id in model_map.items():
        print(f"\n=== Testing {alias} ({repo_id}) ===")
        tm = TextModel(repo_id).to(device)
        batch = tm.tokenizer(
            "The quick brown fox jumps over the lazy dog.",
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(device)
        with torch.no_grad():
            emb = tm(batch)
        print(f"-> {alias} pooled embedding shape: {emb.shape}")
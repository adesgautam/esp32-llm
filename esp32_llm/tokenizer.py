"""
BPE (Byte Pair Encoding) Tokenizer for ESP32LLM.
Trained on the poetry corpus, designed to run both in Python (training/inference) 
and export a decode table for the ESP32 C runtime.

Vocab structure:
  - Tokens 0..N_BASE-1: Individual characters from the base charset
  - Tokens N_BASE..vocab_size-1: Merged subword tokens learned from corpus
"""
import os
import json
import re
from collections import Counter


# Base character set matching the original 44-char vocabulary
BASE_CHARS = list("abcdefghijklmnopqrstuvwxyz0123456789 \n.,'\"-?")


class BPETokenizer:
    def __init__(self, vocab_size: int = 256):
        self.target_vocab_size = vocab_size
        self.base_chars = BASE_CHARS[:]
        self.n_base = len(self.base_chars)
        
        # Vocab: list of token strings, indexed by token ID
        self.vocab = self.base_chars[:]
        # Merges: list of (token_a_id, token_b_id) pairs in learned order
        self.merges = []
        
        # Lookup tables (built after training)
        self.token_to_id = {}
        self.id_to_token = {}
        self._build_lookup()
    
    def _build_lookup(self):
        self.token_to_id = {tok: i for i, tok in enumerate(self.vocab)}
        self.id_to_token = {i: tok for i, tok in enumerate(self.vocab)}
    
    @property
    def vocab_size(self):
        return len(self.vocab)
    
    def _get_pair_counts(self, token_sequences: list) -> Counter:
        """Count adjacent token pair frequencies across all sequences."""
        counts = Counter()
        for seq in token_sequences:
            for i in range(len(seq) - 1):
                counts[(seq[i], seq[i+1])] += 1
        return counts

    def train(self, text: str):
        """Train BPE merges on the given text corpus."""
        print(f"Training BPE tokenizer (target vocab_size={self.target_vocab_size})...")
        
        # Normalize text to our charset
        text = text.lower()
        
        # Initial tokenization: each character -> its base token ID
        char_to_base_id = {ch: i for i, ch in enumerate(self.base_chars)}
        
        # Build flat token array
        tokens = [char_to_base_id[ch] for ch in text if ch in char_to_base_id]
        print(f"  Initial: {len(tokens):,} tokens")
        
        n_merges = self.target_vocab_size - self.n_base
        for merge_i in range(n_merges):
            # Count all adjacent pairs
            pair_counts = Counter()
            for i in range(len(tokens) - 1):
                pair_counts[(tokens[i], tokens[i+1])] += 1
            
            if not pair_counts:
                print(f"  No more pairs to merge at step {merge_i}")
                break
            
            # Find the most frequent pair
            best_pair = pair_counts.most_common(1)[0][0]
            best_count = pair_counts[best_pair]
            
            if best_count < 2:
                print(f"  Stopping early: best pair count = {best_count}")
                break
            
            # Create new token
            new_token_id = len(self.vocab)
            new_token_str = self.vocab[best_pair[0]] + self.vocab[best_pair[1]]
            self.vocab.append(new_token_str)
            self.merges.append(best_pair)
            
            # Replace all occurrences of the pair in the flat array
            new_tokens = []
            i = 0
            while i < len(tokens):
                if i < len(tokens) - 1 and tokens[i] == best_pair[0] and tokens[i+1] == best_pair[1]:
                    new_tokens.append(new_token_id)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens
            
            if (merge_i + 1) % 20 == 0 or merge_i < 5:
                print(f"  Merge {merge_i+1}/{n_merges}: '{repr(self.vocab[best_pair[0]])}' + '{repr(self.vocab[best_pair[1]])}' -> '{repr(new_token_str)}' (count={best_count:,}, tokens={len(tokens):,})")
        
        self._build_lookup()
        print(f"  Final vocab size: {self.vocab_size}")
        print(f"  Total merges learned: {len(self.merges)}")
    
    def encode(self, text: str) -> list:
        """Encode text to token IDs using learned BPE merges."""
        text = text.lower()
        
        # Start with character-level tokens
        char_to_base_id = {ch: i for i, ch in enumerate(self.base_chars)}
        tokens = [char_to_base_id[ch] for ch in text if ch in char_to_base_id]
        
        # Apply merges in order
        for merge_idx, pair in enumerate(self.merges):
            new_token_id = self.n_base + merge_idx
            new_tokens = []
            i = 0
            while i < len(tokens):
                if i < len(tokens) - 1 and tokens[i] == pair[0] and tokens[i+1] == pair[1]:
                    new_tokens.append(new_token_id)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens
        
        return tokens
    
    def decode(self, token_ids: list) -> str:
        """Decode token IDs back to text string."""
        return "".join(self.id_to_token.get(tid, '') for tid in token_ids)
    
    def save(self, path: str):
        """Save tokenizer to JSON."""
        data = {
            "vocab_size": self.vocab_size,
            "n_base": self.n_base,
            "vocab": self.vocab,
            "merges": self.merges,
        }
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"Saved BPE tokenizer to {path}")
    
    def load(self, path: str):
        """Load tokenizer from JSON."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.target_vocab_size = data["vocab_size"]
        self.n_base = data["n_base"]
        self.vocab = data["vocab"]
        self.merges = [tuple(m) for m in data["merges"]]
        self._build_lookup()
        print(f"Loaded BPE tokenizer: vocab_size={self.vocab_size}, merges={len(self.merges)}")

    def export_c_decode_table(self, header_path: str):
        """Export a C header file with the BPE decode lookup table for ESP32."""
        os.makedirs(os.path.dirname(header_path) if os.path.dirname(header_path) else '.', exist_ok=True)
        
        with open(header_path, "w") as f:
            f.write("#ifndef BPE_VOCAB_H\n#define BPE_VOCAB_H\n\n")
            f.write(f"#define BPE_VOCAB_SIZE {self.vocab_size}\n")
            f.write(f"#define BPE_N_BASE {self.n_base}\n\n")
            
            # Store each token as a string literal
            f.write("// BPE vocabulary decode table\n")
            f.write("static const char* const bpe_vocab[] = {\n")
            for i, tok in enumerate(self.vocab):
                # Escape special characters for C string
                escaped = tok.replace("\\", "\\\\").replace('"', '\\"')
                escaped = escaped.replace("\n", "\\n").replace("'", "\\'")
                f.write(f'    "{escaped}",  // {i}\n')
            f.write("};\n\n")
            
            # Store merge rules for encoding on device
            f.write(f"#define BPE_N_MERGES {len(self.merges)}\n\n")
            f.write("// BPE merge rules: pairs of (token_a, token_b) -> new_token_id\n")
            f.write("static const uint16_t bpe_merges[][2] = {\n")
            for a, b in self.merges:
                f.write(f"    {{{a}, {b}}},\n")
            f.write("};\n\n")
            
            f.write("#endif // BPE_VOCAB_H\n")
        
        print(f"Exported C decode table to {header_path}")


if __name__ == "__main__":
    # Quick test
    tok = BPETokenizer(vocab_size=256)
    
    test_text = "pixels glow in late night blue, tracing thoughts of me and you"
    # For standalone test, just train on this small text
    tok.train(test_text * 100)
    
    encoded = tok.encode("pixels glow")
    decoded = tok.decode(encoded)
    print(f"\nTest: 'pixels glow' → {encoded} → '{decoded}'")
    assert decoded == "pixels glow", f"Roundtrip failed: '{decoded}'"
    print("BPE tokenizer roundtrip test passed!")

import os
import re
from datasets import load_dataset
from tqdm import tqdm

def clean_lyrics(text):
    if not isinstance(text, str):
        return ""
    
    # Remove bracketed metadata like [Chorus], [Verse 1], etc.
    text = re.sub(r'\[.*?\]', '', text)
    
    # Replace weird quotes and dashes
    text = text.replace('‘', "'").replace('’', "'")
    text = text.replace('“', '"').replace('”', '"')
    text = text.replace('–', '-').replace('—', '-')
    
    # Strip empty lines
    lines = [line.strip() for line in text.split('\n')]
    lines = [line for line in lines if line]
    
    text = '\n'.join(lines) + '\n\n'
    return text

def main():
    print("Loading dataset from HuggingFace...")
    # juliensimon/autonlp-data-song-lyrics has ~10k english songs.
    # If we need more, we can mix datasets.
    try:
        ds = load_dataset("sebastiandizon/genius-song-lyrics", split="train")
    except Exception as e:
        print(f"Failed to load sebastiandizon dataset: {e}")
        print("Falling back to another lyrics dataset...")
        ds = load_dataset("tsterbak/lyrics-dataset", split="train")
    
    out_dir = os.path.join("datasets", "raw")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "lyrics_corpus_v2.txt")
    
    print(f"Dataset has {len(ds)} songs. Cleaning and exporting to {out_path}...")
    
    total_bytes = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for item in tqdm(ds, desc="Processing"):
            # Depending on dataset, the column might be 'lyrics', 'lyric', 'text'
            text = item.get('lyrics') or item.get('text') or item.get('lyric')
            if not text:
                continue
            
            cleaned = clean_lyrics(text)
            
            # Filter non-ASCII characters to keep within our new A-Z, a-z, punctuation limit
            # Only keep printable ASCII + newline
            cleaned = "".join(c for c in cleaned if (32 <= ord(c) <= 126) or c == '\n')
            
            if len(cleaned) > 50:
                f.write(cleaned)
                total_bytes += len(cleaned)
    
    print(f"\nDone! Exported {total_bytes / 1024 / 1024:.2f} MB of clean lyrics.")

if __name__ == "__main__":
    main()

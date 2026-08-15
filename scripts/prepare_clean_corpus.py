import os
import re
from datasets import load_dataset
from tqdm import tqdm

def clean_and_format_lyrics(text):
    if not isinstance(text, str):
        return ""
    
    # Remove bracketed and parenthesized section headers [Chorus], [Verse], (Bridge), etc.
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'\(Chorus.*?\)', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\(Verse.*?\)', '', text, flags=re.IGNORECASE)
    
    # Standardize punctuation & quotes
    text = text.replace('‘', "'").replace('’', "'")
    text = text.replace('“', '"').replace('”', '"')
    text = text.replace('–', '-').replace('—', '-')
    text = text.replace('…', '...')
    
    # Clean whitespace and strip lines
    lines = [line.strip() for line in text.split('\n')]
    # Remove excessive blank lines and short junk lines (< 2 words)
    valid_lines = []
    for line in lines:
        # Keep ASCII only
        line = "".join(c for c in line if (32 <= ord(c) <= 126))
        if len(line.split()) >= 2:
            valid_lines.append(line.lower())
    
    if len(valid_lines) < 4:
        return ""
    
    return '\n'.join(valid_lines) + '\n\n'

def main():
    print("=" * 60)
    print("  Preparing High-Quality Clean Low-Entropy Corpus")
    print("=" * 60)
    
    try:
        ds = load_dataset("sebastiandizon/genius-song-lyrics", split="train")
    except Exception:
        print("Falling back to lyrics-dataset...")
        ds = load_dataset("tsterbak/lyrics-dataset", split="train")
    
    out_dir = os.path.join("datasets", "raw")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "lyrics_corpus.txt")
    
    print(f"Loaded {len(ds):,} raw items. Filtering and curating...")
    
    total_bytes = 0
    max_target_bytes = 30 * 1024 * 1024  # 30 MB curated clean text
    
    with open(out_path, "w", encoding="utf-8") as f:
        for item in tqdm(ds, desc="Curating"):
            text = item.get('lyrics') or item.get('text') or item.get('lyric')
            if not text:
                continue
            
            cleaned = clean_and_format_lyrics(text)
            if cleaned:
                f.write(cleaned)
                total_bytes += len(cleaned.encode("utf-8"))
                if total_bytes >= max_target_bytes:
                    break
    
    print(f"\n✅ Clean corpus generated: {out_path} ({total_bytes / 1024 / 1024:.2f} MB)")

if __name__ == "__main__":
    main()

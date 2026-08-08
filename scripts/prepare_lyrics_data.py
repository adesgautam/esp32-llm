import os
import re
import requests
from tqdm import tqdm

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9\n\.,\!\?\'\- ]', '', text)
    text = re.sub(r'\n+', '\n', text)
    return text.strip()

def download_file(url, out_path):
    response = requests.get(url, stream=True)
    response.raise_for_status()
    total_size = int(response.headers.get('content-length', 0))
    with open(out_path, 'wb') as f, tqdm(
        desc=out_path,
        total=total_size,
        unit='B',
        unit_scale=True
    ) as pbar:
        for data in response.iter_content(chunk_size=1024):
            size = f.write(data)
            pbar.update(size)

def main():
    print("Fetching Kanye West Hip-Hop Lyrics from GitHub...")
    os.makedirs('datasets/raw', exist_ok=True)
    
    download_url = "https://raw.githubusercontent.com/robbiebarrat/rapping-neural-network/master/lyrics.txt"
    temp_file = "datasets/raw/temp_lyrics.txt"
    
    print(f"Downloading {download_url}...")
    try:
        download_file(download_url, temp_file)
    except Exception as e:
        print(f"Failed to download: {e}")
        return

    out_path = 'datasets/raw/lyrics_corpus.txt'
    total_chars = 0
    
    print("Cleaning lyrics...")
    with open(temp_file, 'r', encoding='utf-8', errors='ignore') as infile:
        raw_text = infile.read()
        
    clean_lyrics = clean_text(raw_text)
    
    with open(out_path, 'w', encoding='utf-8') as outfile:
        outfile.write(clean_lyrics)
        total_chars = len(clean_lyrics)
                
    # Clean up temp file
    os.remove(temp_file)
    
    print(f"\nDone! Extracted {total_chars / 1024 / 1024:.2f} MB of clean hip-hop lyrics.")
    print(f"Corpus saved to: {out_path}")

if __name__ == '__main__':
    main()

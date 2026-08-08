import os
import re
import glob
import pandas as pd
from tqdm import tqdm

def clean_text(text):
    text = str(text).lower()
    # Keep alphanumeric, newlines, commas, single quotes, double quotes, basic punctuation
    text = re.sub(r'[^a-z0-9\n\.,\!\?\'" \-]', '', text)
    text = re.sub(r'\n+', '\n', text)
    return text.strip()

def main():
    print("Aggregating lyrics from local datasets (data1, data2, data3)...")
    out_path = 'datasets/raw/lyrics_corpus.txt'
    total_chars = 0
    target_chars = 10 * 1024 * 1024  # 10 MB limit
    
    with open(out_path, 'w', encoding='utf-8') as outfile:
        
        # 1. Parse data1/*.txt
        print("Processing data1/*.txt...")
        txt_files = glob.glob('datasets/raw/data1/*.txt')
        for f in tqdm(txt_files):
            try:
                with open(f, 'r', encoding='utf-8', errors='ignore') as infile:
                    clean_lyrics = clean_text(infile.read())
                    if len(clean_lyrics) > 50:
                        outfile.write(clean_lyrics + "\n\n")
                        total_chars += len(clean_lyrics) + 2
            except Exception as e:
                print(f"Error reading {f}: {e}")
            if total_chars >= target_chars:
                break
                
        # 2. Parse data2/lyrics_raw.csv (column: 'raw_lyrics')
        if total_chars < target_chars and os.path.exists('datasets/raw/data2/lyrics_raw.csv'):
            print("Processing data2/lyrics_raw.csv...")
            try:
                df2 = pd.read_csv('datasets/raw/data2/lyrics_raw.csv')
                for lyric in tqdm(df2['raw_lyrics'].dropna()):
                    clean_lyrics = clean_text(lyric)
                    if len(clean_lyrics) > 50:
                        outfile.write(clean_lyrics + "\n\n")
                        total_chars += len(clean_lyrics) + 2
                    if total_chars >= target_chars:
                        break
            except Exception as e:
                print(f"Error reading data2: {e}")

        # 3. Parse data3/updated_rappers.csv (column: 'lyric')
        if total_chars < target_chars and os.path.exists('datasets/raw/data3/updated_rappers.csv'):
            print("Processing data3/updated_rappers.csv...")
            try:
                df3 = pd.read_csv('datasets/raw/data3/updated_rappers.csv')
                for lyric in tqdm(df3['lyric'].dropna()):
                    clean_lyrics = clean_text(lyric)
                    if len(clean_lyrics) > 50:
                        outfile.write(clean_lyrics + "\n\n")
                        total_chars += len(clean_lyrics) + 2
                    if total_chars >= target_chars:
                        break
            except Exception as e:
                print(f"Error reading data3: {e}")

    print(f"\nDone! Extracted {total_chars / 1024 / 1024:.2f} MB of combined hip-hop lyrics.")
    print(f"Corpus saved to: {out_path}")

if __name__ == '__main__':
    main()

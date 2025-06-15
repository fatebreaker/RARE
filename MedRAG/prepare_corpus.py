import os
import json
import pickle
import argparse
import tqdm

corpus_names = {
    "PubMed": ["pubmed"],
    "Textbooks": ["textbooks"],
    "StatPearls": ["statpearls"],
    "Wikipedia": ["wikipedia"],
    "MedCorp": ["pubmed", "textbooks", "statpearls", "wikipedia"],
}


def download_corpus(corpus, db_dir="./corpus"):
    """Download a specific corpus if it doesn't exist"""
    corpus_path = os.path.join(db_dir, corpus)
    chunk_dir = os.path.join(corpus_path, "chunk")

    if not os.path.exists(chunk_dir):
        print(f"Cloning the {corpus} corpus from Huggingface...")
        os.makedirs(db_dir, exist_ok=True)
        os.system(f"git clone https://huggingface.co/datasets/MedRAG/{corpus} {corpus_path}")

        if corpus == "statpearls":
            print("Downloading the statpearls corpus from NCBI bookshelf...")
            os.system(
                f"wget https://ftp.ncbi.nlm.nih.gov/pub/litarch/3d/12/statpearls_NBK430685.tar.gz --no-check-certificate -P {corpus_path}")
            os.system(f"tar -xzvf {os.path.join(corpus_path, 'statpearls_NBK430685.tar.gz')} -C {corpus_path}")
            print("Chunking the statpearls corpus...")
            os.system("python src/data/statpearls.py")
    else:
        print(f"Corpus {corpus} already exists at {chunk_dir}")

    return chunk_dir


def extract_chunks_from_corpus(corpus, db_dir="./corpus"):
    """Extract all chunks from a single corpus"""
    chunk_dir = download_corpus(corpus, db_dir)

    if not os.path.exists(chunk_dir):
        print(f"Error: Chunk directory {chunk_dir} does not exist!")
        return []

    print(f"Extracting chunks from {corpus}...")
    all_chunks = []

    # Get all .jsonl files in the chunk directory
    fnames = sorted([fname for fname in os.listdir(chunk_dir) if fname.endswith(".jsonl")])

    for fname in tqdm.tqdm(fnames, desc=f"Processing {corpus}"):
        fpath = os.path.join(chunk_dir, fname)

        # Skip empty files
        if not os.path.exists(fpath) or os.path.getsize(fpath) == 0:
            continue

        try:
            with open(fpath, 'r', encoding='utf-8') as f:
                content = f.read().strip()

            if content == "":
                continue

            # Parse each line as JSON
            for line in content.split('\n'):
                if line.strip():
                    try:
                        item = json.loads(line)
                        # Combine title and content
                        chunk_text = item["title"] + ": " + item["content"]
                        all_chunks.append(chunk_text)
                    except json.JSONDecodeError as e:
                        print(f"Warning: Failed to parse JSON in {fname}: {e}")
                        continue

        except Exception as e:
            print(f"Error processing file {fpath}: {e}")
            continue

    print(f"Extracted {len(all_chunks)} chunks from {corpus}")
    return all_chunks


def extract_and_save_chunks(corpus_name, db_dir="./corpus", output_dir="./"):
    """Main function to extract chunks and save to pickle file"""

    if corpus_name not in corpus_names:
        print(f"Error: Unknown corpus name '{corpus_name}'")
        print(f"Available corpus names: {list(corpus_names.keys())}")
        return

    print(f"Processing corpus: {corpus_name}")
    print(f"Sub-corpora: {corpus_names[corpus_name]}")

    all_chunks = []

    # Process each sub-corpus
    for corpus in corpus_names[corpus_name]:
        chunks = extract_chunks_from_corpus(corpus, db_dir)
        all_chunks.extend(chunks)

    # Save to pickle file
    output_file = os.path.join(output_dir, f"{corpus_name}_chunks.pickle")

    print(f"Saving {len(all_chunks)} chunks to {output_file}...")
    with open(output_file, 'wb') as f:
        pickle.dump(all_chunks, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Successfully saved {len(all_chunks)} chunks to {output_file}")
    return output_file


def main():
    parser = argparse.ArgumentParser(description="Extract chunks from medical corpus and save to pickle file")
    parser.add_argument("corpus_name", type=str,
                        help=f"Name of the corpus to process. Options: {list(corpus_names.keys())}")
    parser.add_argument("--db_dir", type=str, default="./corpus",
                        help="Directory to store downloaded corpora (default: ./corpus)")
    parser.add_argument("--output_dir", type=str, default="./",
                        help="Directory to save the output pickle file (default: ./)")

    args = parser.parse_args()

    # Create directories if they don't exist
    os.makedirs(args.db_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    # Extract and save chunks
    output_file = extract_and_save_chunks(args.corpus_name, args.db_dir, args.output_dir)

    if output_file:
        print("\n" + "=" * 50)
        print("SUMMARY:")
        print(f"Corpus: {args.corpus_name}")
        print(f"Output file: {output_file}")

        # Load and verify the saved file
        try:
            with open(output_file, 'rb') as f:
                chunks = pickle.load(f)
            print(f"Verified: {len(chunks)} chunks saved successfully")

            # Show a sample chunk
            if chunks:
                print(f"Sample chunk: {chunks[0][:200]}...")

        except Exception as e:
            print(f"Error verifying saved file: {e}")


if __name__ == "__main__":
    main()
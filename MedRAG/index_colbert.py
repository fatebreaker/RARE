#!/usr/bin/env python3
"""
ColBERT Indexer Script

A script to create ColBERT indices from medical corpus collections.
Takes an index name and collection pickle file as input arguments.
"""

import argparse
import os
import pickle
import sys
from pathlib import Path

import colbert
from colbert import Indexer, Searcher
from colbert.infra import Run, RunConfig, ColBERTConfig
from colbert.data import Queries, Collection


def load_collection(collection_file):
    """
    Load collection from pickle file.

    Args:
        collection_file (str): Path to the pickle file containing the collection

    Returns:
        list: Collection of documents
    """
    if not os.path.exists(collection_file):
        raise FileNotFoundError(f"Collection file not found: {collection_file}")

    print(f"Loading collection from: {collection_file}")

    try:
        with open(collection_file, 'rb') as file:
            collection = pickle.load(file)

        print(f"Successfully loaded {len(collection)} documents")
        return collection

    except Exception as e:
        raise Exception(f"Error loading collection file: {e}")


def create_colbert_index(index_name, collection_file, checkpoint, output_dir,
                         nbits=2, doc_maxlen=300, kmeans_niters=4, nranks=1):
    """
    Create a ColBERT index from a collection.

    Args:
        index_name (str): Name for the index
        collection_file (str): Path to collection pickle file
        checkpoint (str): Path to ColBERT checkpoint
        output_dir (str): Directory to save the index
        nbits (int): Number of bits for encoding (default: 2)
        doc_maxlen (int): Maximum document length in tokens (default: 300)
        kmeans_niters (int): Number of k-means iterations (default: 4)
        nranks (int): Number of GPUs to use (default: 1)
    """

    # Load collection
    collection = load_collection(collection_file)

    # Validate checkpoint
    if not os.path.exists(checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    print(f"Creating ColBERT index:")
    print(f"  Index name: {index_name}")
    print(f"  Collection size: {len(collection)} documents")
    print(f"  Checkpoint: {checkpoint}")
    print(f"  Output directory: {output_dir}")
    print(f"  Document max length: {doc_maxlen} tokens")
    print(f"  Encoding bits: {nbits}")
    print(f"  K-means iterations: {kmeans_niters}")
    print(f"  Number of GPUs: {nranks}")

    # Create ColBERT index
    with Run().context(RunConfig(nranks=nranks, experiment='indexing', root=output_dir)):
        config = ColBERTConfig(
            doc_maxlen=doc_maxlen,
            nbits=nbits,
            kmeans_niters=kmeans_niters
        )

        indexer = Indexer(checkpoint=checkpoint, config=config)

        print("Starting indexing process...")
        indexer.index(name=index_name, collection=collection, overwrite='resume')

        print(f"Index '{index_name}' created successfully!")


def validate_args(args):
    """Validate command line arguments."""

    # Check collection file
    if not os.path.exists(args.collection_file):
        print(f"Error: Collection file not found: {args.collection_file}")
        sys.exit(1)

    # Check checkpoint
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint not found: {args.checkpoint}")
        sys.exit(1)

    # Validate index name (should be filesystem-safe)
    invalid_chars = ['/', '\\', ':', '*', '?', '"', '<', '>', '|']
    if any(char in args.index_name for char in invalid_chars):
        print(f"Error: Index name contains invalid characters: {args.index_name}")
        sys.exit(1)

    # Check if output directory is writable
    try:
        os.makedirs(args.output_dir, exist_ok=True)
    except PermissionError:
        print(f"Error: Cannot create output directory: {args.output_dir}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Create ColBERT index from a medical corpus collection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Index Wikipedia corpus
  python colbert_indexer.py wikipedia Wikipedia_chunks.pickle

  # Index MedCorp with custom settings
  python colbert_indexer.py medcorp MedCorp_chunks.pickle \\
    --checkpoint ./models/colbert_checkpoint \\
    --output_dir ./indices \\
    --doc_maxlen 512 \\
    --nbits 4

  # Index with multiple GPUs
  python colbert_indexer.py pubmed PubMed_chunks.pickle --nranks 4
        """
    )

    # Required arguments
    parser.add_argument(
        'index_name',
        type=str,
        help='Name for the ColBERT index (will be used as directory name)'
    )

    parser.add_argument(
        'collection_file',
        type=str,
        help='Path to the pickle file containing the document collection'
    )

    # Optional arguments
    parser.add_argument(
        '--checkpoint',
        type=str,
        default='/home/htran/generation/MedRAG/checkpoint/colbertv2.0',
        help='Path to ColBERT checkpoint directory (default: %(default)s)'
    )

    parser.add_argument(
        '--output_dir',
        type=str,
        default='./colbert_indices',
        help='Directory to save the index (default: %(default)s)'
    )

    parser.add_argument(
        '--nbits',
        type=int,
        default=2,
        choices=[1, 2, 4, 8],
        help='Number of bits for encoding each dimension (default: %(default)s)'
    )

    parser.add_argument(
        '--doc_maxlen',
        type=int,
        default=300,
        help='Maximum document length in tokens (default: %(default)s)'
    )

    parser.add_argument(
        '--kmeans_niters',
        type=int,
        default=4,
        help='Number of k-means clustering iterations (default: %(default)s)'
    )

    parser.add_argument(
        '--nranks',
        type=int,
        default=1,
        help='Number of GPUs to use for indexing (default: %(default)s)'
    )

    parser.add_argument(
        '--dry_run',
        action='store_true',
        help='Show what would be done without actually creating the index'
    )

    args = parser.parse_args()

    # Validate arguments
    validate_args(args)

    if args.dry_run:
        print("DRY RUN - Would create index with the following settings:")
        print(f"  Index name: {args.index_name}")
        print(f"  Collection file: {args.collection_file}")
        print(f"  Checkpoint: {args.checkpoint}")
        print(f"  Output directory: {args.output_dir}")
        print(f"  Document max length: {args.doc_maxlen}")
        print(f"  Encoding bits: {args.nbits}")
        print(f"  K-means iterations: {args.kmeans_niters}")
        print(f"  Number of GPUs: {args.nranks}")

        # Load collection to show size
        try:
            collection = load_collection(args.collection_file)
            print(f"  Collection size: {len(collection)} documents")
        except Exception as e:
            print(f"  Error loading collection: {e}")

        return

    # Create the index
    try:
        create_colbert_index(
            index_name=args.index_name,
            collection_file=args.collection_file,
            checkpoint=args.checkpoint,
            output_dir=args.output_dir,
            nbits=args.nbits,
            doc_maxlen=args.doc_maxlen,
            kmeans_niters=args.kmeans_niters,
            nranks=args.nranks
        )

        print("\n" + "=" * 50)
        print("INDEXING COMPLETED SUCCESSFULLY!")
        print("=" * 50)
        print(f"Index location: {os.path.join(args.output_dir, args.index_name)}")

    except Exception as e:
        print(f"\nError during indexing: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
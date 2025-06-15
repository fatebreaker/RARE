#!/usr/bin/env python3
"""
ColBERT Search API

A Flask API server for searching through ColBERT indices.
Provides REST endpoints for semantic search over document collections.
"""

import argparse
import math
import os
import pickle
import sys
from functools import lru_cache
from pathlib import Path

from flask import Flask, jsonify, request, render_template
from dotenv import load_dotenv

from colbert.infra import Run, RunConfig, ColBERTConfig
from colbert import Searcher


class ColBERTSearchAPI:
    """ColBERT Search API handler"""

    def __init__(self, index_dir, collection_file=None, max_cache_size=1000000):
        """
        Initialize the search API.

        Args:
            index_dir (str): Path to ColBERT index directory
            collection_file (str): Path to collection pickle file (optional)
            max_cache_size (int): Maximum cache size for search results
        """
        self.index_dir = index_dir
        self.collection_file = collection_file
        self.max_cache_size = max_cache_size
        self.collection = None
        self.searcher = None
        self.stats = {"api_calls": 0, "cache_hits": 0, "cache_misses": 0}

        self._load_searcher()
        self._setup_cache()

    def _load_searcher(self):
        """Load the ColBERT searcher and collection."""

        # Validate index directory
        if not os.path.exists(self.index_dir):
            raise FileNotFoundError(f"Index directory not found: {self.index_dir}")

        print(f"Loading ColBERT index from: {self.index_dir}")

        # Load collection if provided
        if self.collection_file:
            if not os.path.exists(self.collection_file):
                raise FileNotFoundError(f"Collection file not found: {self.collection_file}")

            print(f"Loading collection from: {self.collection_file}")
            with open(self.collection_file, 'rb') as file:
                self.collection = pickle.load(file)
            print(f"Loaded {len(self.collection)} documents")

        # Initialize searcher
        try:
            if self.collection:
                self.searcher = Searcher(
                    index=self.index_dir,
                    index_root=self.index_dir,
                    collection=self.collection
                )
            else:
                self.searcher = Searcher(index=self.index_dir, index_root=self.index_dir)

            print("ColBERT searcher initialized successfully")

        except Exception as e:
            raise Exception(f"Failed to initialize searcher: {e}")

    def _setup_cache(self):
        """Setup LRU cache for search results."""

        @lru_cache(maxsize=self.max_cache_size)
        def _cached_search(query, k):
            self.stats["cache_misses"] += 1
            return self._search_internal(query, k)

        self._cached_search = _cached_search

    def _search_internal(self, query, k):
        """Internal search method."""

        if not query or not query.strip():
            return {"error": "Empty query provided"}

        k = min(int(k), 100)  # Limit to max 100 results

        try:
            # Perform search
            pids, ranks, scores = self.searcher.search(query.strip(), k=100)
            pids, ranks, scores = pids[:k], ranks[:k], scores[:k]

            # Calculate probabilities
            probs = [math.exp(score) for score in scores]
            prob_sum = sum(probs)
            if prob_sum > 0:
                probs = [prob / prob_sum for prob in probs]

            # Format results
            topk = []
            for pid, rank, score, prob in zip(pids, ranks, scores, probs):
                try:
                    # Get text from searcher's collection
                    text = self.searcher.collection[pid]

                    result = {
                        'text': text,
                        'pid': int(pid),
                        'rank': int(rank),
                        'score': float(score),
                        'prob': float(prob)
                    }

                    # Add passage from original collection if available
                    if self.collection and pid < len(self.collection):
                        result['passage'] = self.collection[pid]

                    topk.append(result)

                except (IndexError, KeyError) as e:
                    print(f"Warning: Could not retrieve document {pid}: {e}")
                    continue

            # Sort by score (descending) then by pid (ascending)
            topk = sorted(topk, key=lambda x: (-x['score'], x['pid']))

            return {
                "query": query,
                "num_results": len(topk),
                "topk": topk
            }

        except Exception as e:
            return {"error": f"Search failed: {str(e)}"}

    def search(self, query, k=10):
        """
        Search the collection.

        Args:
            query (str): Search query
            k (int): Number of results to return

        Returns:
            dict: Search results
        """

        if k is None:
            k = 10

        self.stats["api_calls"] += 1

        # Try to get from cache first
        cache_key = (query, k)
        try:
            result = self._cached_search(query, k)
            if cache_key in self._cached_search.cache_info()._asdict():
                self.stats["cache_hits"] += 1
            return result
        except Exception as e:
            return {"error": f"Search error: {str(e)}"}

    def get_stats(self):
        """Get API statistics."""
        cache_info = self._cached_search.cache_info()

        return {
            "api_calls": self.stats["api_calls"],
            "cache_hits": cache_info.hits,
            "cache_misses": cache_info.misses,
            "cache_size": cache_info.currsize,
            "cache_max_size": cache_info.maxsize,
            "hit_rate": cache_info.hits / max(1, cache_info.hits + cache_info.misses)
        }


def create_app(index_dir, collection_file=None, max_cache_size=1000000):
    """Create Flask application."""

    app = Flask(__name__)

    # Initialize search API
    try:
        search_api = ColBERTSearchAPI(
            index_dir=index_dir,
            collection_file=collection_file,
            max_cache_size=max_cache_size
        )
    except Exception as e:
        print(f"Error initializing search API: {e}")
        sys.exit(1)

    @app.route("/", methods=["GET"])
    def index():
        """API documentation page."""
        return jsonify({
            "name": "ColBERT Search API",
            "version": "1.0",
            "endpoints": {
                "/api/search": {
                    "method": "GET",
                    "parameters": {
                        "query": "Search query (required)",
                        "k": "Number of results (optional, default: 10, max: 100)"
                    },
                    "example": "/api/search?query=diabetes+symptoms&k=5"
                },
                "/api/stats": {
                    "method": "GET",
                    "description": "Get API usage statistics"
                },
                "/health": {
                    "method": "GET",
                    "description": "Health check endpoint"
                }
            },
            "index_dir": index_dir,
            "collection_file": collection_file
        })

    @app.route("/api/search", methods=["GET"])
    def api_search():
        """Search endpoint."""

        if request.method != "GET":
            return jsonify({"error": "Method not allowed"}), 405

        query = request.args.get("query")
        k = request.args.get("k", 10)

        if not query:
            return jsonify({"error": "Query parameter is required"}), 400

        try:
            k = int(k)
            if k <= 0:
                return jsonify({"error": "k must be positive"}), 400
        except ValueError:
            return jsonify({"error": "k must be an integer"}), 400

        result = search_api.search(query, k)

        if "error" in result:
            return jsonify(result), 500

        return jsonify(result)

    @app.route("/api/stats", methods=["GET"])
    def api_stats():
        """Statistics endpoint."""
        return jsonify(search_api.get_stats())

    @app.route("/health", methods=["GET"])
    def health_check():
        """Health check endpoint."""
        return jsonify({
            "status": "healthy",
            "searcher_loaded": search_api.searcher is not None,
            "collection_loaded": search_api.collection is not None
        })

    @app.errorhandler(404)
    def not_found(error):
        return jsonify({"error": "Endpoint not found"}), 404

    @app.errorhandler(500)
    def internal_error(error):
        return jsonify({"error": "Internal server error"}), 500

    return app


def main():
    parser = argparse.ArgumentParser(
        description="ColBERT Search API Server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python colbert_api.py ./colbert_indices/medcorp

  # With collection file  
  python colbert_api.py ./colbert_indices/medcorp --collection ./chunks/MedCorp_chunks.pickle

  # Custom host and port
  python colbert_api.py ./colbert_indices/medcorp --host 127.0.0.1 --port 8080

  # Production mode
  python colbert_api.py ./colbert_indices/medcorp --production
        """
    )

    # Required arguments
    parser.add_argument(
        'index_dir',
        type=str,
        help='Path to the ColBERT index directory'
    )

    # Optional arguments
    parser.add_argument(
        '--collection',
        type=str,
        help='Path to collection pickle file (optional, for enhanced results)'
    )

    parser.add_argument(
        '--host',
        type=str,
        default='0.0.0.0',
        help='Host to bind the server to (default: 0.0.0.0)'
    )

    parser.add_argument(
        '--port',
        type=int,
        default=6000,
        help='Port to bind the server to (default: 6000)'
    )

    parser.add_argument(
        '--cache_size',
        type=int,
        default=1000000,
        help='Maximum cache size for search results (default: 1000000)'
    )

    parser.add_argument(
        '--debug',
        action='store_true',
        help='Run in debug mode'
    )

    parser.add_argument(
        '--production',
        action='store_true',
        help='Run in production mode (disables debug)'
    )

    args = parser.parse_args()

    # Load environment variables
    load_dotenv()

    # Validate arguments
    if not os.path.exists(args.index_dir):
        print(f"Error: Index directory not found: {args.index_dir}")
        sys.exit(1)

    if args.collection and not os.path.exists(args.collection):
        print(f"Error: Collection file not found: {args.collection}")
        sys.exit(1)

    # Create Flask app
    app = create_app(
        index_dir=args.index_dir,
        collection_file=args.collection,
        max_cache_size=args.cache_size
    )

    # Configure debug mode
    debug_mode = args.debug and not args.production

    print(f"Starting ColBERT Search API server...")
    print(f"Index directory: {args.index_dir}")
    print(f"Collection file: {args.collection or 'None'}")
    print(f"Host: {args.host}")
    print(f"Port: {args.port}")
    print(f"Debug mode: {debug_mode}")
    print(f"Cache size: {args.cache_size}")
    print(f"\nAPI will be available at: http://{args.host}:{args.port}")
    print(f"Search endpoint: http://{args.host}:{args.port}/api/search?query=your+query")

    # Run the server
    app.run(
        host=args.host,
        port=args.port,
        debug=debug_mode,
        threaded=True
    )


if __name__ == "__main__":
    main()
import colbert
from colbert import Indexer, Searcher
from colbert.infra import Run, RunConfig, ColBERTConfig
from colbert.data import Queries, Collection
import pickle
from datasets import load_dataset

# dataset = 'lifestyle'
# datasplit = 'dev'
#
# collection_dataset = load_dataset("colbertv2/lotte_passages", dataset)
# collection = [x['text'] for x in collection_dataset[datasplit + '_collection']]
#
# queries_dataset = load_dataset("colbertv2/lotte", dataset)
# queries = [x['query'] for x in queries_dataset['search_' + datasplit]]
#
# print('Loaded ', len(queries),' queries and ', len(collection),' passages')
#
# print(queries[24])
# print()
# print(collection[19929])
# print()
#
#
# max_id = 1000
#
# index_name = f'{dataset}.{datasplit}.{nbits}bits'
#
# answer_pids = [x['answers']['answer_pids'] for x in queries_dataset['search_' + datasplit]]
# filtered_queries = [q for q, apids in zip(queries, answer_pids) if any(x < max_id for x in apids)]
#
# print(f'Filtered down to {len(filtered_queries)} queries')
#
#
# # To create the searcher using its relative name (i.e., not a full path), set
# # experiment=value_used_for_indexing in the RunConfig.
# with Run().context(RunConfig(experiment='notebook')):
#     searcher = Searcher(index=index_name, collection=collection)
#
# query = filtered_queries[0] # try with an in-range query or supply your own
# print(f"#> {query}")
#
# # Find the top-3 passages for this query
# results = searcher.search(query, k=3)
#
# # Print out the top-k retrieved passages
# for passage_id, passage_rank, passage_score in zip(*results):
#     print(f"\t [{passage_rank}] \t\t {passage_score:.1f} \t\t {searcher.collection[passage_id]}")

if __name__ == "__main__":
    checkpoint = '/home/htran/generation/MedRAG/checkpoint/colbertv2.0'
    nbits = 2  # encode each dimension with 2 bits
    doc_maxlen = 300  # truncate passages at 300 tokens
    # index_name = "wikipedia"
    # with open('/data/data_user_alpha/MedRAG/corpus/wikipedia/all_chunks.pickle', 'rb') as file:
    #     collection = pickle.load(file)
    index_name = "textstat"
    with open('corpus/textstat_chunks.pickle', 'rb') as file:
        collection = pickle.load(file)
    with Run().context(RunConfig(nranks=1, experiment='notebook', root="corpus/experiments")):  # nranks specifies the number of GPUs to use
        config = ColBERTConfig(doc_maxlen=doc_maxlen, nbits=nbits, kmeans_niters=4) # kmeans_niters specifies the number of iterations of k-means clustering; 4 is a good and fast default.
                                                                                    # Consider larger numbers for small datasets.
        indexer = Indexer(checkpoint=checkpoint, config=config)
        indexer.index(name=index_name, collection=collection, overwrite=False)

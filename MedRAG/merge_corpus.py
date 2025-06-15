import pickle

all_chunks = []
# Load the list from the file
with open('/data/data_user_alpha/MedRAG/corpus/textbooks/all_chunks.pickle', 'rb') as file:
    textbooks = pickle.load(file)

with open('/data/data_user_alpha/MedRAG/corpus/statpearls/all_chunks.pickle', 'rb') as file:
    stat = pickle.load(file)

# with open('corpus/pubmed/all_chunks.pickle', 'rb') as file:
#     pubmed = pickle.load(file)
#
# with open('corpus/wikipedia/all_chunks.pickle', 'rb') as file:
#     wiki = pickle.load(file)


all_chunks.extend(textbooks)
all_chunks.extend(stat)
# all_chunks.extend(pubmed)
# all_chunks.extend(wiki)
print(len(all_chunks))

with open('corpus/textstat_chunks.pickle', 'wb') as file:
    pickle.dump(all_chunks, file, protocol=pickle.HIGHEST_PROTOCOL)


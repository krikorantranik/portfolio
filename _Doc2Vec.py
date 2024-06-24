import numpy as np
import pandas as pd
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import nltk
from nltk.tokenize import word_tokenize

#train and save
tagged_data = []
for index, row in trainset.iterrows():
 part = TaggedDocument(words=word_tokenize(row[0]), tags=[str(index)])
 tagged_data.append(part)
model = Doc2Vec(vector_size=350, min_count=3, epochs=50, window=10, dm=1)
model.build_vocab(tagged_data)
model.train(tagged_data, total_examples=model.corpus_count, epochs=model.epochs)
model.save("d2v.model")
print("Model Saved")


#vectorize
a = []
for index, row in maindataset.iterrows():
 nlptext = row['NLPtext']
 ids = row['index']
 vector = model.infer_vector(word_tokenize(nlptext))
 vector = pd.DataFrame(vector).T
 vector.index = [ids]
 a.append(vector)
textvectors = pd.concat(a)
textvectors


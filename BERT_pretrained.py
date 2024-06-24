#
import pandas as pd
from sentence_transformers import SentenceTransformer

#download model
bertmodel = SentenceTransformer('all-mpnet-base-v2')

#create embdedding from pandas column
reviews_embedding = bertmodel.encode(maindataset["NLPtext"])

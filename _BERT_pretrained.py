#
import pandas as pd
from sentence_transformers import SentenceTransformer
from bertopic import BERTopic

#download model
bertmodel = SentenceTransformer('all-mpnet-base-v2')

#create embdedding from pandas column
reviews_embedding = bertmodel.encode(maindataset["NLPtext"])

#Bertopic
model = BERTopic(verbose=True,embedding_model='bert-base-uncased', min_topic_size= 3)
headline_topics, _ = model.fit_transform(moviestbl["text"])
moviestbl["topic"] = headline_topics
topicinfo = model.get_topic_info()

#toptopics
topic = 0
print("topic: " + str(topic))
print(model.get_topic(topic))
print(moviestbl[moviestbl['topic']==topic]['Title'])
topic = 1
print("topic: " + str(topic))
print(model.get_topic(topic))
print(moviestbl[moviestbl['topic']==topic]['Title'])
topic = 2
print("topic: " + str(topic))
print(model.get_topic(topic))
print(moviestbl[moviestbl['topic']==topic]['Title'])
topic = 3
print("topic: " + str(topic))
print(model.get_topic(topic))
print(moviestbl[moviestbl['topic']==topic]['Title'])
topic = 4
print("topic: " + str(topic))
print(model.get_topic(topic))
print(moviestbl[moviestbl['topic']==topic]['Title'])
topic = 5
print("topic: " + str(topic))
print(model.get_topic(topic))
print(moviestbl[moviestbl['topic']==topic]['Title'])
topic = 6
print("topic: " + str(topic))
print(model.get_topic(topic))
print(moviestbl[moviestbl['topic']==topic]['Title'])
topic = 7
print("topic: " + str(topic))
print(model.get_topic(topic))
print(moviestbl[moviestbl['topic']==topic]['Title'])
topic = 8
print("topic: " + str(topic))
print(model.get_topic(topic))
print(moviestbl[moviestbl['topic']==topic]['Title'])

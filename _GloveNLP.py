

#download pretrained models
model1 = api.load("glove-wiki-gigaword-200")
model2 = api.load("glove-twitter-100")

#most similar words from both models, and average them out. See the results of a test
def close_words(wrd):
 tbl1 = pd.DataFrame(model1.most_similar(wrd, topn=10), columns=['word','siml']).set_index('word')
 tbl2 = pd.DataFrame(model2.most_similar(wrd, topn=10), columns=['word','siml']).set_index('word')
 tbl = pd.concat((tbl1, tbl2), axis=1).mean(axis=1)
 tbl = pd.DataFrame(tbl).sort_values(0, ascending=False)
 return tbl

#search similar words
search = 'barrels'
wordsim = close_words(search)

#lematize
lemmatizer = WordNetLemmatizer()
wordsim['Lemma'] = wordsim.index
wordsim['Lemma'] = wordsim['Lemma'].apply(lambda x: lemmatizer.lemmatize(x))



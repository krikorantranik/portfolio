import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

#create ruless
rules_m = apriori(newset, min_support = 0.045, use_colnames = True)
rules = association_rules(rules_m, metric ="lift")
rules = rules.sort_values(['confidence'], ascending =[False])

#merge in a line
rules['antecedents'] = rules['antecedents'].apply(lambda x: list(x))
rules['consequents'] = rules['consequents'].apply(lambda x: list(x))
rules['lenantecedents'] = rules['antecedents'].apply(lambda x: len(x))
rules['lenconsequents'] = rules['consequents'].apply(lambda x: len(x))

# clean and filter
rules = rules[rules['lenantecedents']>1]
rules['ListAntecedents'] = rules['antecedents'].apply(lambda x: '(' + ','.join(x) + ')')
rules['ListConsequents'] = rules['consequents'].apply(lambda x: '(' + ','.join(x) + ')')
search = rules[['antecedents']]
search = search.explode(['antecedents'])
search = search.pivot_table(index=search.index, columns='antecedents', aggfunc=len, fill_value=0)
search['Total'] = search.sum(axis = 1)

def get_out(airport1, airport2, onlyExact, minSupport, minConfidence, minLift, numRec):
 idx = search[search[airport1]==1]
 idx = idx[idx[airport2]==1]
 if onlyExact == True:
  idx = idx[idx['Total']==2]
 out = rules.loc[idx.index]
 out = out[out['confidence']>=minConfidence]
 out = out[out['support']>=minSupport]
 out = out[out['lift']>=minLift]
 out = out[out['lenconsequents']<=numRec]
 out = out.sort_values('confidence', ascending=False)
 return out[['antecedents','consequents','support','confidence','lift']]

result = get_out(airport1 = 'AMS', airport2 = 'FRA', onlyExact = True, minSupport = 0.05, minConfidence = 0.1, minLift = 1, numRec=1)
result

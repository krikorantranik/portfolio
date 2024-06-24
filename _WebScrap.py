import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
pd.options.mode.chained_assignment = None

#simple website scrap
url = 'https://en.wikipedia.org/wiki/List_of_best-selling_films_in_the_United_States'
tbls = pd.read_html(url)
tbls[0]

#extract urls from page
url = 'https://en.wikipedia.org/wiki/List_of_best-selling_films_in_the_United_States'
reqs = requests.get(url)
soup = BeautifulSoup(reqs.text, 'html.parser')
urls = []
for link in soup.find_all('a'):
 row = pd.DataFrame({'Name': [link.get('title')], 'link': [link.get('href')]})
 urls.append(row)
urls = pd.concat(urls, axis=0)
urls

#scrap from hyperlink list / table
def scraptext(txt):
 reqs = requests.get(txt)
 soup = BeautifulSoup(reqs.text, 'html.parser')
 ps = []
 for par in soup.find_all('p'):
  ps.append(par.text)
 ps = ' '.join(ps)
 return ps
moviestbl["text"] = moviestbl["link"].apply(lambda x: scraptext(x))
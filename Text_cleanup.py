import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
from io import StringIO
from html.parser import HTMLParser
import re

#load data
maindataset = pd.read_csv("Restaurant_reviews.csv")
maindataset

#remove tags
class MLStripper(HTMLParser):
    def __init__(self):
        super().__init__()
        self.reset()
        self.strict = False
        self.convert_charrefs= True
        self.text = StringIO()
    def handle_data(self, d):
        self.text.write(d)
    def get_data(self):
        return self.text.getvalue()

def strip_tags(html):
    s = MLStripper()
    s.feed(html)
    return s.get_data()

#clean text
def preprepare(eingang):
 ausgang = strip_tags(eingang)
 ausgang = eingang.lower()
 ausgang = ausgang.replace(u'\xa0', u' ')
 ausgang = re.sub(r'^\s*$',' ',str(ausgang))
 ausgang = ausgang.replace('|', ' ')
 ausgang = ausgang.replace('ï', ' ')
 ausgang = ausgang.replace('»', ' ')
 ausgang = ausgang.replace('¿', '. ')
 ausgang = ausgang.replace('ï»¿', ' ')
 ausgang = ausgang.replace('"', ' ')
 ausgang = ausgang.replace("'", " ")
 ausgang = ausgang.replace('?', ' ')
 ausgang = ausgang.replace('!', ' ')
 ausgang = ausgang.replace(',', ' ')
 ausgang = ausgang.replace(';', ' ')
 ausgang = ausgang.replace('.', ' ')
 ausgang = ausgang.replace("(", " ")
 ausgang = ausgang.replace(")", " ")
 ausgang = ausgang.replace("{", " ")
 ausgang = ausgang.replace("}", " ")
 ausgang = ausgang.replace("[", " ")
 ausgang = ausgang.replace("]", " ")
 ausgang = ausgang.replace("~", " ")
 ausgang = ausgang.replace("@", " ")
 ausgang = ausgang.replace("#", " ")
 ausgang = ausgang.replace("$", " ")
 ausgang = ausgang.replace("%", " ")
 ausgang = ausgang.replace("^", " ")
 ausgang = ausgang.replace("&", " ")
 ausgang = ausgang.replace("*", " ")
 ausgang = ausgang.replace("<", " ")
 ausgang = ausgang.replace(">", " ")
 ausgang = ausgang.replace("/", " ")
 ausgang = ausgang.replace("\\", " ")
 ausgang = ausgang.replace("`", " ")
 ausgang = ausgang.replace("+", " ")
 ausgang = ausgang.replace("=", " ")
 ausgang = ausgang.replace("_", " ")
 ausgang = ausgang.replace("-", " ")
 ausgang = ausgang.replace(':', ' ')
 ausgang = ausgang.replace('\n', ' ').replace('\r', ' ')
 ausgang = ausgang.replace(" +", " ")
 ausgang = ausgang.replace(" +", " ")
 ausgang = ausgang.replace('?', ' ')
 ausgang = re.sub('[^a-zA-Z]', ' ', ausgang)
 ausgang = re.sub(' +', ' ', ausgang)
 ausgang = re.sub('\ +', ' ', ausgang)
 ausgang = re.sub(r'\s([?.!"](?:\s|$))', r'\1', ausgang)
 return ausgang

maindataset["NLPtext"] = maindataset["Review"]
maindataset["NLPtext"] = maindataset["NLPtext"].str.lower()
maindataset["NLPtext"] = maindataset["NLPtext"].apply(lambda x: preprepare(str(x)))



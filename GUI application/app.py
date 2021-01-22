from flask import Flask, render_template, request, redirect, url_for
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')
from nltk.tokenize import TreebankWordTokenizer
from nltk.stem.porter import *
import numpy
from math import log,sqrt
import pandas as pd
import timeit
import pickle
from tabulate import tabulate

#model and query stuff
def tokenize_sentence(sentence):    # Tokenize a string and remove punctuations
#arguments : string
    sentence=sentence.lower()
    tokenizer=TreebankWordTokenizer()
    tokens_list_with_punct = tokenizer.tokenize(sentence.lower())
    tokens_list_without_punct=[]
    for x in tokens_list_with_punct:
        if x.isalpha():
            tokens_list_without_punct.append(x)
    return tokens_list_without_punct

def remove_stopwords(words):        # Remove english stop words from a string
#arguments : List of strings
    stopwords_list=stopwords.words('english')
    filtered_words=[]
    for x in words:
        if x not in stopwords_list:
            filtered_words.append(x)
    return filtered_words

def stem_words(words_list):         # Stem words in a string
#arguments : List of strings
    ps=PorterStemmer()
    stemmed_words=[]
    for x in range(len(words_list)):
        stemmed_words.append(ps.stem(words_list[x]))
    return stemmed_words

def preprocess_sentence(sentence):  # Method to call above defined preprocessing tasks on a string
#arguments : string
    return stem_words(remove_stopwords(tokenize_sentence(sentence)))

# actual model definition
class IR_model():
 def __init__(self):
# constructor method
  self.__word_dict={}      #dictionary of all words in corpus.Each unique word is mapped to an index
  self.__word_dict_size=0  #size of word_dict
  self.__doc_list_size=0   #number of docs in corpus
  self.__score_matrix=None #2D matrix to store tf-idf score
 
 def __addDocument(self,document): #adds document to the matrix and store tf in cells
 # arguments : corpus as a list of tuples ; No return type. 
  for word in document[1]:
   self.__score_matrix[self.__doc_list_size][self.__word_dict[word]]+=1
  self.__doc_list_size+=1

 def build_Vector_Space(self,documents): # computes values for cells in vector space
 #arguments : corpus as a list of tuples ; No return type. 
  for document in documents:             # assign index to unique words
   for word in document[1]:
    if word not in self.__word_dict:
     self.__word_dict[word]=self.__word_dict_size
     self.__word_dict_size+=1

  self.__score_matrix=numpy.zeros((len(documents),self.__word_dict_size))
        
  for document in documents:              # assign tf value to cells
   self.__addDocument(document)

  idf=numpy.zeros((self.__word_dict_size))
  df=numpy.zeros((self.__word_dict_size))

  for word in self.__word_dict:            # calculate df of all words in corpus
   x=self.__word_dict[word]
   for i in range(len(self.__score_matrix)):
    if self.__score_matrix[i][x]!=0:
     df[x]+=1

  for i in range(self.__word_dict_size):   # calculate idf of all words in corpus
   idf[i]=log(len(self.__score_matrix)/df[i])

  for i in range(len(self.__score_matrix)):# fill cells with tf-idf score
   for j in range(self.__word_dict_size):
    if self.__score_matrix[i][j]!=0:
     self.__score_matrix[i][j]=(1+log(self.__score_matrix[i][j]))*(idf[j])
        
 def Search(self,query,documents): # Finds the tf_idf score of query for all docs in corpus and returns it
#arguments : user query as list of terms, corpus as list of tuples
#Return type : list of tuples
  query_df=numpy.zeros((self.__word_dict_size))
  query_idf=numpy.zeros((self.__word_dict_size))
  for word in query:               # for each word in query, find df and idf
   if word in self.__word_dict:
    if query_df[self.__word_dict[word]]==0:
     for document in documents:
      if word in document[1]:
       query_df[self.__word_dict[word]]+=1
       query_idf[self.__word_dict[word]]=log(len(self.__score_matrix)/query_df[self.__word_dict[word]])

  query_score=[]  # To store score between query and each document
  for document in documents:  # Finding tf for each term in query for each doc and find score
   query_tf=numpy.zeros(self.__word_dict_size)
   score=0
   for word in query:
    if word in self.__word_dict:
     if query_tf[self.__word_dict[word]]==0:
      for term in document[1]:
       if term==word:
        query_tf[self.__word_dict[word]]+=1
      if query_tf[self.__word_dict[word]] > 0 :
       query_tf[self.__word_dict[word]]=1+log(query_tf[self.__word_dict[word]])
       score+= query_tf[self.__word_dict[word]] * query_idf[self.__word_dict[word]] 
                                              # score(query,doc)=sum of score for all terms in query and doc
   query_score.append([(score,document[0])]) # append doc id and score for returning
   
  return query_score 

#loading dataset and pickled files
df = pd.read_csv("Song.csv", error_bad_lines=False)

pickle_in = open("docs.pickle", "rb")
docs = pickle.load(pickle_in)
pickle_in.close()

pickle_in = open("model.pickle", "rb")
Model = pickle.load(pickle_in)
pickle_in.close()


def requestResults(name):
    start = timeit.default_timer()

    # Preprocess query text
    q = name
    q=preprocess_sentence(q)

    # Search for query in model
    f=Model.Search(q,docs)

    # arrange/select top10 results
    isallzeros = 1
    f.sort(reverse=True)
    top10=[]
    for i in range(10):
      for j in f[i]:
          if j[0] > 0:
            isallzeros = 0
    if isallzeros == 1 :
        return tabulate([' '], headers=['Sorry no match found'], tablefmt='html')
        #return 'Sorry, no match found'
    for i in range(10):
      for j in f[i]:
        top10.append(j[1])

    ans=[]
    count = 1

    for i in top10:
        ans.append([count, df.iloc[i]['Artist Name'],df.iloc[i]['Song Name'],df.iloc[i]['Clean Lyrics']])
        count += 1

    table = ans

    stop = timeit.default_timer()
    return  '<body style="background-color: #E0FFFF;">' + "Top10 Results in: " + '<b>' + str(stop-start) + "</b>" + " seconds <br><br>" + tabulate(table, headers=['Rank','Artist Name','Song Name','Lyrics'], tablefmt='html') + "</body>"



# start Flask
app = Flask(__name__)

# render default home webpage
@app.route('/')
def hello():
    return render_template('home.html')

#when post method detected, redirect to success function
@app.route('/', methods=['POST', 'GET'])
def get_data():
    if request.method == 'POST':
        user = request.form['search']
        return redirect(url_for('success', name=user))

#get the data/results for requested query
@app.route('/success/<name>')
def success(name):
    return str(requestResults(name))

if __name__ == '__main__':
    app.run(debug=True)


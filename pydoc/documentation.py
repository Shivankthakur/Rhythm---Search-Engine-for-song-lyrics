def tokenize_sentence(sentence): 

'''
  Tokenize a string and remove punctuations
  
  Parameters:
  sentence(string)
  
  Returns:
  tokens_list_without_punct(list of strings)
'''
    sentence=sentence.lower()
    tokenizer=TreebankWordTokenizer()
    tokens_list_with_punct = tokenizer.tokenize(sentence.lower())
    tokens_list_without_punct=[]
    for x in tokens_list_with_punct:
        if x.isalpha():
            tokens_list_without_punct.append(x)
    return tokens_list_without_punct

def remove_stopwords(words):    

'''
  Remove english stop words from a string
  
  Parameters:
  words(list of string)
  
  Returns:
  filtered_words(list of strings)
'''
    stopwords_list=stopwords.words('english')
    filtered_words=[]
    for x in words:
        if x not in stopwords_list:
            filtered_words.append(x)
    return filtered_words

def stem_words(words_list): 

'''
  Stem words in a string
  
  Parameters:
  words_list(list of string)
  
  Returns:
  stemmed_words(list of strings)
'''
    ps=PorterStemmer()
    stemmed_words=[]
    for x in range(len(words_list)):
        stemmed_words.append(ps.stem(words_list[x]))
    return stemmed_words

def preprocess_sentence(sentence):

'''
  Method to call above defined preprocessing tasks on a string
  
  Parameters:
  sentence(string)
  
  Returns:
  stem_words(remove_stopwords(tokenize_sentence(sentence)))(list of strings)
'''
    return stem_words(remove_stopwords(tokenize_sentence(sentence)))


class IR_model():
  """
    A class to represent IR model.

    ...

    Attributes
    ----------
    word_dict : dictionary
      dictionary of all words in corpus.Each unique word is mapped to an index
    word_dict_size : int
      size of word_dict
    doc_list_size : int
      number of docs in corpus
    score_matrix : 2D floating point array
      2D array to store tf.idf values
    
    Methods
    -------
    addDocument(self,document):
      adds document to the matrix and store tf in cells
    build_Vector_Space(self,documents):
      computes values for cells in vector space
    Search(self,query,documents):
      Finds the tf_idf score of query for all docs in corpus and returns it
  
  """
 def __init__(self):

      """
        Constructs all the necessary attributes for the IR model object.

        Parameters
        ----------
        None
      """


  self.__word_dict={}      
  self.__word_dict_size=0  
  self.__doc_list_size=0  
  self.__score_matrix=None 
 
 def __addDocument(self,document): 
 
        """
        adds document to the matrix and store tf in cells

        Parameters
        ----------
        document : list
              list of tuples whose 1st element in tuple is id and 2nd is list of strings
            
        """
  for word in document[1]:
   self.__score_matrix[self.__doc_list_size][self.__word_dict[word]]+=1
  self.__doc_list_size+=1

 def build_Vector_Space(self,documents): 
 
        """
        computes values for cells in vector space

        Parameters
        ----------
        documents : list
              list of list of tuples whose 1st element in tuple is id and 2nd is list of strings
            
        """  
  for document in documents:          
   for word in document[1]:
    if word not in self.__word_dict:
     self.__word_dict[word]=self.__word_dict_size
     self.__word_dict_size+=1

  self.__score_matrix=numpy.zeros((len(documents),self.__word_dict_size))
        
  for document in documents:             
   self.__addDocument(document)

  idf=numpy.zeros((self.__word_dict_size))
  df=numpy.zeros((self.__word_dict_size))

  for word in self.__word_dict:          
   x=self.__word_dict[word]
   for i in range(len(self.__score_matrix)):
    if self.__score_matrix[i][x]!=0:
     df[x]+=1

  for i in range(self.__word_dict_size):   
   idf[i]=log(len(self.__score_matrix)/df[i])

  for i in range(len(self.__score_matrix)):
   for j in range(self.__word_dict_size):
    if self.__score_matrix[i][j]!=0:
     self.__score_matrix[i][j]=(1+log(self.__score_matrix[i][j]))*(idf[j])
        
 def Search(self,query,documents): 

        """
        Finds the tf_idf score of query for all docs in corpus and returns it

        Parameters
        ----------
        query : list
          user query as list of terms
        documents : list
          list of list of tuples whose 1st element in tuple is id and 2nd is list of strings
        Returns
        -------
        query_score : list
          list containing tf.idf scores
        """
  query_df=numpy.zeros((self.__word_dict_size))
  query_idf=numpy.zeros((self.__word_dict_size))
  for word in query:               #
   if word in self.__word_dict:
    if query_df[self.__word_dict[word]]==0:
     for document in documents:
      if word in document[1]:
       query_df[self.__word_dict[word]]+=1
       query_idf[self.__word_dict[word]]=log(len(self.__score_matrix)/query_df[self.__word_dict[word]])

  query_score=[]  
  for document in documents:  
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
                                            
   query_score.append([(score,document[0])]) 
   
  return query_score 

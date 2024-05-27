
# If you haven't installed it, install the nltk package
# conda install nltk

import nltk
# download the example texts
nltk.download('gutenberg')
nltk.download('genesis')
nltk.download('inaugural')
nltk.download('nps_chat')
nltk.download('webtext')
nltk.download('treebank')
from nltk.book import *
from nltk import word_tokenize

# display text1
text1

# open the book as text file
f = open('moby.txt')
raw = f.read()

tokens = word_tokenize(raw)

#-----------------------------------------------------------------------------
# Some basic string/nltk commands to search and navigate text data
#-----------------------------------------------------------------------------

# show concordances of the word Moby
text1.concordance("Moby")

# show words that appear arround the word captain
text1.similar("captain")

# now find similar terms around captain and sea
text1.common_contexts(["captain", "sea"])

# produce a dispersion plot for the following three words
text1.dispersion_plot(["captain", "Moby", "whale"])

# counting words
len(text1)

# generating an alphabetic list of all words in the text
sorted(set(text3))

# referencing individual words
text1[3707]

# or finding their location
text1.index('captain')


#-----------------------------------------------------------------------------
# Cleaning and preparing text data
#-----------------------------------------------------------------------------

# retrieve the list of stopwords (English) from nltk
from nltk.corpus import stopwords

# have a look which words are included
print(stopwords.words('english'))

# safe the list of stop words
stopwords = nltk.corpus.stopwords.words('english')

# let's see how many words in Moby Dick are NOT stop words
def content_fraction(text):
    stopwords = nltk.corpus.stopwords.words('english')
    content = [w for w in text if w.lower() not in stopwords]
    return len(content) / len(text)


content_fraction(text1)


# show the 20 most common words
fdist1 = FreqDist(text1)
fdist1.most_common(20)

# to remove the stopwords,
text1_ns = [word for word in text1 if word.lower() not in stopwords]

# now show the most common 20 words again
fdist1 = FreqDist(text1_ns)
fdist1.most_common(20)


#-----------------------------------------------------------------------------
# Producing word clouds
#-----------------------------------------------------------------------------

from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
stopwords = set(STOPWORDS)

def show_wordcloud(data, title = None):
    wordcloud = WordCloud(
        background_color='white',
        stopwords=stopwords,
        max_words=200,
        max_font_size=40, 
        scale=3,
        random_state=1 # chosen at random by flipping a coin; it was heads
    ).generate(str(data))

    fig = plt.figure(1, figsize=(12, 12))
    plt.axis('off')
    if title: 
        fig.suptitle(title, fontsize=20)
        fig.subplots_adjust(top=2.3)

    plt.imshow(wordcloud)
    plt.show()

show_wordcloud(text1_ns)


#-----------------------------------------------------------------------------
# Classification of text
#-----------------------------------------------------------------------------

# build a function that extracts the last letter of a word
def gender_features(word):
    return {'last_letter': word[-1]}

# import the data and the random library
from nltk.corpus import names
import random

# these are our labeled names in alphabetical order
labeled_names = ([(name, 'male') for name in names.words('male.txt')] + [(name, 'female') for name in names.words('female.txt')])

# shuffle them to get some randomness here
random.shuffle(labeled_names)

# collect the features
featuresets = [(gender_features(n), gender) for (n, gender) in labeled_names]

# split the data into test and training
train_set, test_set = featuresets[500:], featuresets[:500]

# load the bayes classifier
classifier = nltk.NaiveBayesClassifier.train(train_set)

# apply it to new names
classifier.classify(gender_features('Neo'))
classifier.classify(gender_features('Trinity'))

# show accuracy
print(nltk.classify.accuracy(classifier, test_set))

# most informative features
classifier.show_most_informative_features(5)


#-----------------------------------------------------------------------------
# Topic analysis
#-----------------------------------------------------------------------------

import pandas as pd
import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer, SnowballStemmer, PorterStemmer
from nltk.stem.porter import *
import numpy as np
np.random.seed(2018)
import nltk
nltk.download('wordnet')

# load the data
data = pd.read_csv('abcnews-date-text.csv', error_bad_lines=False);

# store the headline text
data_text = data[['headline_text']]

# and assign an index to each headline
data_text['index'] = data_text.index
documents = data_text

# see how many docs there are
len(documents)


# define two functions to lemmatize (not covered in class) and stemming for preprocessing
stemmer = PorterStemmer()
def lemmatize_stemming(text):
    return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))

def preprocess(text):
    result = []
    for token in gensim.utils.simple_preprocess(text):
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
            result.append(lemmatize_stemming(token))
    return result

# look at a sample document to see what the preprocessing did
doc_sample = documents[documents['index'] == 4310].values[0][0]
print('original document: ')
words = []

for word in doc_sample.split(' '):
    words.append(word)
print(words)

print("Stemmed document")
print(preprocess(doc_sample))

# now apply this to all the headlines
processed_docs = documents['headline_text'].map(preprocess)
processed_docs[:10]

# now generate a bag of words
dictionary = gensim.corpora.Dictionary(processed_docs)

count = 0
for k, v in dictionary.iteritems():
    print(k, v)
    count += 1
    if count > 10:
        break
    
# keep the 100,000 most frequent words
dictionary.filter_extremes(no_below=15, no_above=0.5, keep_n=100000)

# create a list of words and their frequency with which they appear within and across documents
bow_corpus = [dictionary.doc2bow(doc) for doc in processed_docs]

from gensim import corpora, models

tfidf = models.TfidfModel(bow_corpus)
corpus_tfidf = tfidf[bow_corpus]

from pprint import pprint

for doc in corpus_tfidf:
    pprint(doc)
    break

# I shorten the computational time a bit by only taking a subsample
mycorpus = bow_corpus[0:100000]

# run the LDA model
lda_model = gensim.models.LdaMulticore(mycorpus, num_topics=10, id2word=dictionary, workers=4)



#-----------------------------------------------------------------------------
# Sentiment analysis
#-----------------------------------------------------------------------------

nltk.download('twitter_samples')
nltk.download('averaged_perceptron_tagger')

from nltk.corpus import twitter_samples
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tag import pos_tag
import nltk

# positive and negative tweets
positive_tweets = twitter_samples.strings('positive_tweets.json')
negative_tweets = twitter_samples.strings('negative_tweets.json')

# text without sentiment
text = twitter_samples.strings('tweets.20150430-223406.json')


# tokenize the tweets
tweet_tokens = twitter_samples.tokenized('positive_tweets.json')[0]

# lemmatize the tweets
def lemmatize_sentence(tokens):
    lemmatizer = WordNetLemmatizer()
    lemmatized_sentence = []
    for word, tag in pos_tag(tokens):
        if tag.startswith('NN'):
            pos = 'n'
        elif tag.startswith('VB'):
            pos = 'v'
        else:
            pos = 'a'
        lemmatized_sentence.append(lemmatizer.lemmatize(word, pos))
    return lemmatized_sentence

print(lemmatize_sentence(tweet_tokens[0]))

# remove noise from the data such as hyperlinks
import re, string

def remove_noise(tweet_tokens, stop_words = ()):

    cleaned_tokens = []

    for token, tag in pos_tag(tweet_tokens):
        token = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+#]|[!*\(\),]|'\
                       '(?:%[0-9a-fA-F][0-9a-fA-F]))+','', token)
        token = re.sub("(@[A-Za-z0-9_]+)","", token)

        if tag.startswith("NN"):
            pos = 'n'
        elif tag.startswith('VB'):
            pos = 'v'
        else:
            pos = 'a'

        lemmatizer = WordNetLemmatizer()
        token = lemmatizer.lemmatize(token, pos)

        if len(token) > 0 and token not in string.punctuation and token.lower() not in stop_words:
            cleaned_tokens.append(token.lower())
    return cleaned_tokens


# remove stop words
from nltk.corpus import stopwords
stop_words = stopwords.words('english')

print(remove_noise(tweet_tokens[0], stop_words))

# save the cleaned data in separate objects
positive_tweet_tokens = twitter_samples.tokenized('positive_tweets.json')
negative_tweet_tokens = twitter_samples.tokenized('negative_tweets.json')

positive_cleaned_tokens_list = []
negative_cleaned_tokens_list = []

for tokens in positive_tweet_tokens:
    positive_cleaned_tokens_list.append(remove_noise(tokens, stop_words))

for tokens in negative_tweet_tokens:
    negative_cleaned_tokens_list.append(remove_noise(tokens, stop_words))
    
# compute the word densities
def get_all_words(cleaned_tokens_list):
    for tokens in cleaned_tokens_list:
        for token in tokens:
            yield token

all_pos_words = get_all_words(positive_cleaned_tokens_list)

from nltk import FreqDist

freq_dist_pos = FreqDist(all_pos_words)
print(freq_dist_pos.most_common(10))


# now get the tweets for the model
def get_tweets_for_model(cleaned_tokens_list):
    for tweet_tokens in cleaned_tokens_list:
        yield dict([token, True] for token in tweet_tokens)

positive_tokens_for_model = get_tweets_for_model(positive_cleaned_tokens_list)
negative_tokens_for_model = get_tweets_for_model(negative_cleaned_tokens_list)

# split data into test and training
import random

positive_dataset = [(tweet_dict, "Positive")
                     for tweet_dict in positive_tokens_for_model]

negative_dataset = [(tweet_dict, "Negative")
                     for tweet_dict in negative_tokens_for_model]

dataset = positive_dataset + negative_dataset

random.shuffle(dataset)

train_data = dataset[:7000]
test_data = dataset[7000:]

# apply the model
from nltk import classify
from nltk import NaiveBayesClassifier
classifier = NaiveBayesClassifier.train(train_data)

print("Accuracy is:", classify.accuracy(classifier, test_data))

print(classifier.show_most_informative_features(10))

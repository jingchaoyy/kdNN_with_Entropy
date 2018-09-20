from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words
from nltk.stem.porter import PorterStemmer
from gensim import corpora, models
import gensim
import random
import math


def randomData(num=100, catNum=5, catRange=10):
    # num: Num of Entry
    # catNum: max Number of categories for each entry
    # catRange: types of categories
    allCategories = []
    # catSum = list(string.ascii_uppercase)
    for i in range(num):
        catTmp = set()
        while len(catTmp) < catNum:
            catTmp.add(str(random.randint(0, catRange)))
        tmp = ""
        for j in catTmp:
            tmp = tmp + ', ' + j
        allCategories.append(tmp[2:])
    # print(allPoints)
    # print(allCategories)
    return allCategories


def LDA(trainSet, testSet, topics=5, times=20):
    # topics: # of topics in the result
    # times: # of passes during training
    tokenizer = RegexpTokenizer(r'\w+')

    # create English stop words list
    en_stop = get_stop_words('en')

    # Create p_stemmer of class PorterStemmer
    p_stemmer = PorterStemmer()

    # create sample documents
    # r1 = "1, 2, 3"
    # r2 = "2, 3, 4"
    # r3 = "1, 3, 5"
    # r4 = "2, 4, 5"
    # r5 = "1, 5, 6"

    # compile sample documents into a list
    # r_set = [r1, r2, r3, r4, r5]

    # print r_set

    # list for tokenized documents in loop
    texts = []

    # loop through document list
    for i in trainSet:
        # clean and tokenize document string
        raw = i.lower()
        tokens = tokenizer.tokenize(raw)
        # remove stop words from tokens
        stopped_tokens = [i for i in tokens if not i in en_stop]
        # stem tokens
        stemmed_tokens = [p_stemmer.stem(i) for i in stopped_tokens]
        # add tokens to list
        texts.append(stemmed_tokens)

    # turn our tokenized documents into a id <-> term dictionary
    dictionary = corpora.Dictionary(texts)

    # convert tokenized documents into a document-term matrix
    corpus = [dictionary.doc2bow(text) for text in texts]

    # generate LDA model
    ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=topics, id2word=dictionary, passes=times)
    # print(ldamodel.print_topics(num_topics, num_words=3))

    # unseen_document = ['k1','k2','k3']
    bow_vector = corpora.Dictionary(texts).doc2bow(testSet)

    for index, score in sorted(ldamodel[bow_vector], key=lambda tup: -1 * tup[1]):
        print("Score: {}\t Topic: {}".format(score, ldamodel.print_topic(index, 3)))


def main():
    data = randomData()
    LDA(data, ['1', '3', '5', '7', '9'])


if __name__ == "__main__":
    main()
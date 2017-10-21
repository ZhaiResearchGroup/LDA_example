import sys
import argparse
# import email-parse.parse_email
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation as LDA
import numpy as np
import json
import numpy as np
import matplotlib.pyplot as plt
from stop_words import get_stop_words
np.set_printoptions(threshold=100000, linewidth=200)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-ddir', nargs='?', help='directory to emails')
    parser.add_argument('-n_topics', nargs='?', default=4, type=int, help='number of topics')
    parser.add_argument('-n_words', nargs='?', default=10, type=int, help='number of words to display per topic')
    parser.add_argument('-t', action="store_true", help='run on test data')
    parser.add_argument('-filter_stop_words', action="store_true", help='run on test data')
    args = parser.parse_args()
    print args

    if not args.t:
        # get list of JSON emails
        emails = parse_email(args.ddir)
    else:
        with open('json.json') as f:
            emails = json.load(f);

    # extract messages
    documents = []

    if args.filter_stop_words:
        stop_words = get_stop_words('en')

        for email in emails:
            message = email["message"]
            for w in stop_words:
                message = message.replace(w, '')
            documents.append(message)
    else:
        for email in emails:
            documents.append(email["message"])

    # create the document-word matrix (n_documents X n_words)
    vectorizer = CountVectorizer()
    vectorizer.fit(documents)
    M_docword = vectorizer.transform(documents)

    word_to_index = vectorizer.vocabulary_
    index_to_word = np.chararray(len(word_to_index), itemsize=100)
    for word in word_to_index:
        index = word_to_index[word]
        index_to_word[index] = word

    # fit LDA
    lda = LDA(n_topics=args.n_topics)
    lda.fit(M_docword)

    # p(word|topic)  (n_topics X n_words)
    w_z = lda.components_ / lda.components_.sum(axis=1)[:, np.newaxis]
    # p(topic|document)  (n_documents X n_topics)
    z_d = lda.transform(M_docword)

    print "w_z\n", w_z
    print "\nz_d\n", z_d
    print "\n"

    top_word_args = np.argsort(w_z, axis=1)[:,-1*args.n_words:]
    top_words = np.chararray((w_z.shape[0], args.n_words, 2), itemsize=100)
    for i in xrange(args.n_topics):
        top_words[i,:,0] = index_to_word[top_word_args[i]]
        top_words[i,:,1] = w_z[i,top_word_args[i]]

    print top_words

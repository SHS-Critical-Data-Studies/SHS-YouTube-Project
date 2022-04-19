import re
import nltk
from gensim.corpora import Dictionary
from nltk import WordNetLemmatizer
from nltk.corpus import stopwords
en_stopwords = set(stopwords.words('english'))
en_stopwords.update([s.capitalize() for s in stopwords.words('english')])


def preprocess(comment, tag=False):
    """
    Preprocess un commentaire :
        - supprime les nombres
        - passe la première lettre de chaque phrase en minuscule
        - supprime les stopwords et mots de 1 lettre
        - passe en minuscules les mots tout en majuscules
        - applique une lemmatization
        - supprime à nouveau les stopwords et mots de 1 lettre
    :param comment: string contenant le commentaire
    :param tag: whether we process a tag
    :return:
    """
    # Supprime les nombres
    comment = re.sub(r'\d+', '', comment)

    # Première lettre de chaque phrase en minuscule
    lower_first_word = lambda tab: ' '.join(tab[0].lower() + tab[1:])
    if not tag:
        comment = ' '.join([lower_first_word(sentence.split(' ')) for sentence in comment.split('.')])
    else:
        # There is no sentences in tags
        comment = comment.lower()

    # Tokenize par mot
    tokenizer = nltk.RegexpTokenizer(r'\w+')
    words_tokens = tokenizer.tokenize(comment)

    # Supprime les stopwords et mots de 1 lettre
    remove_stopwords = lambda wts: [w for w in wts if (not w in en_stopwords) and len(w) > 1]
    words_tokens = remove_stopwords(words_tokens)

    # Passe les mots en majuscules en minuscules
    for i in range(len(words_tokens)):
        if words_tokens[i].isupper():
            words_tokens[i] = words_tokens[i].lower()

    # Lemmatization avec WordNet
    lemmatizer = WordNetLemmatizer()
    words_tokens = [lemmatizer.lemmatize(wt) for wt in words_tokens]

    # Supprime les stopwords et mots de 1 lettre
    words_tokens = remove_stopwords(words_tokens)

    return words_tokens


def preprocessor(comment):
    """
    Preprocess un commentaire:
        - supprime les nombres
        - passe la première lettre de chaque phrase en minuscule
    :param comment: commentaire à traiter
    :return: commentaire traité
    """
    # Supprime les nombres
    comment = re.sub(r'\d+', '', comment)

    # Première lettre de chaque phrase en minuscule
    lower_first_word = lambda tab: ' '.join([tab[0].lower()] + tab[1:])
    comment = ' '.join([lower_first_word(sentence.split(' ')) for sentence in comment.split('.')])

    return comment


def tokenizer(comment):
    """
    Tokenize et process les tokens d'un commentaire
    :param comment: commentaire à traiter
    :return: list des tokens pour ce commentaire
    """
    # Tokenize par mot
    tokenizer = nltk.RegexpTokenizer(r'\w+')
    words_tokens = tokenizer.tokenize(comment)

    # Supprime les stopwords et mots de 1 lettre
    remove_stopwords = lambda wts: [w for w in wts if (not w in en_stopwords) and len(w) > 1]
    words_tokens = remove_stopwords(words_tokens)

    # Passe les mots en majuscules en minuscules
    for i in range(len(words_tokens)):
        if words_tokens[i].isupper():
            words_tokens[i] = words_tokens[i].lower()

    # Lemmatization avec WordNet
    lemmatizer = WordNetLemmatizer()
    words_tokens = [lemmatizer.lemmatize(wt) for wt in words_tokens]

    # Supprime les stopwords et mots de 1 lettre
    words_tokens = remove_stopwords(words_tokens)

    return words_tokens


def create_corpus(docs, min_wordcount, max_freq):
    """
    Creates a dictionary and corpus from the documents
    :param docs: documents
    :param min_wordcount: minimum number of documents a word needs to appear in to be kept
    :param max_freq: maximum percentage of documents a word should appear in the be kept
    :return: dictionary and corpus
    """
    dictionary = Dictionary(docs)

    # Filter out words that are in less than min_wordcount documents and that appear in more than max_freq documents
    dictionary.filter_extremes(no_below=min_wordcount, no_above=max_freq)
    dictionary.compactify()

    # Bag-of-words representation of the documents
    corpus = [dictionary.doc2bow(doc) for doc in docs]

    return dictionary, corpus
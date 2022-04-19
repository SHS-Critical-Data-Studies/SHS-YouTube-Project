import re
import nltk
from nltk import WordNetLemmatizer
from nltk.corpus import stopwords
en_stopwords = set(stopwords.words('english'))
en_stopwords.update([s.capitalize() for s in stopwords.words('english')])


def preprocess(comment):
    """
    Preprocess un commentaire :
        - supprime les nombres
        - passe la première lettre de chaque phrase en minuscule
        - supprime les stopwords et mots de 1 lettre
        - passe en minuscules les mots tout en majuscules
        - applique une lemmatization
        - supprime à nouveau les stopwords et mots de 1 lettre
    :param comment: string contenant le commentaire
    :return:
    """
    # Supprime les nombres
    comment = re.sub(r'\d+', '', comment)

    # Première lettre de chaque phrase en minuscule
    lower_first_word = lambda tab: ' '.join(tab[0].lower() + tab[1:])
    comment = ' '.join([lower_first_word(sentence.split(' ')) for sentence in comment.split('.')])

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

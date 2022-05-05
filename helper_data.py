import numpy as np
import os
import pandas as pd
import spacy
from spacy.language import Language
from spacy_langdetect import LanguageDetector


@Language.factory('language_detector')
def language_detector(nlp, name):
    return LanguageDetector()


nlp = spacy.load("en_core_web_sm")
nlp.add_pipe('language_detector', last=True)


def load_walk(path, mode):
    """
    Load the data of a walk
    :param path: path of the file
    :param mode: what data to load
    :return: dataframe containing the data of the walk
    """
    use_cols = None
    if mode == 'comments':
        use_cols = range(1, 6)
    elif mode == 'infos':
        use_cols = range(1, 12)
    else:
        print('Mode not supported')
        return
    return pd.read_csv(path, compression='bz2', usecols=use_cols)


def load_from_folder(folder, processing, mode):
    """
    Load the comments of all the walks
    present in the given folder
    :param folder: folder containing the walks
    :param processing: function that process each file of
    the folder
    :param mode: what data to load
    :return: dataframe containing all the comments
    """
    dfs = []
    walks = 0

    with os.scandir(folder) as it:
        for entry in it:
            if entry.name.endswith(f'{mode}.csv.bz2') and entry.is_file():
                df = processing(os.path.join(folder, entry.name), walks)
                dfs.append(df)
                walks += 1

    return pd.concat(dfs)


def load_all_walks_comments(folder, keep_en):
    """
    Load the comments of all the walks
    present in the given folder
    :param folder: folder containing the walks
    :param keep_en: only keeps english comments
    :return: dataframe containing all the comments
    """

    def processing(path, walk):
        df = load_walk(path, 'comments')
        if keep_en:
            df = df.loc[df['text'].apply(lambda x: nlp(str(x))._.language['language'] == 'en'), :]
        df['walk'] = walk
        return df

    return load_from_folder(folder, processing, 'comments')


def load_all_walks_tags(folder, keep_en):
    """
    Load the tags of all the walks
    present in the given folder
    :param folder: folder containing the walks
    :param keep_en: only keeps english tags
    :return: dataframe containing all the tags
    """

    def processing(path, walk):
        df = load_walk(path, 'infos')

        def process(tags):
            res = []
            for t in tags[1:-1].replace("'", '').split(', '):
                if keep_en:
                    if nlp(str(t))._.language['language'] == 'en':
                        res.append(t)
                else:
                    res.append(t)

            return ' '.join(res) if len(res) > 0 else np.nan

        df['keywords'] = df['keywords'].apply(process)
        df['walk'] = walk
        return df

    return load_from_folder(folder, processing, 'infos')


def load_all_walks_infos(folder, keep_en):
    """
    Load the tags of all the walks
    present in the given folder
    :param folder: folder containing the walks
    :param keep_en: only keeps english tags
    :return: dataframe containing all the tags
    """

    def processing(path, walk):
        df = load_walk(path, 'infos')

        def process(tags):
            res = []
            for t in tags[1:-1].replace("'", '').split(', '):
                if keep_en:
                    if nlp(str(t))._.language['language'] == 'en':
                        res.append(t)
                else:
                    res.append(t)

            return ' '.join(res) if len(res) > 0 else np.nan

        df['keywords'] = df['keywords'].apply(process)
        df['walk'] = walk
        return df

    return load_from_folder(folder, processing, 'infos')
import collections
import logging
import subprocess
import sys
import argparse

import csv
import re
import functools

from importlib import resources

from transformers import pipeline


"""
Classif sentences along various dimensions.

Currently computes subjectivity, sentiment based on a transformer model; concreteness simply by averaging word scores from a lookup table (Brysbaert et al.) 

Example:
$ sentclass "This is a sentence" --subj --conc

"""

# TODO: Rename to qtype


_subjectivity_models = {
    'en': 'GroNLP/mdebertav3-subjectivity-multilingual',    # also for ge, tu, ar
    # 'en': "cffl/bert-base-styleclassification-subjective-neutral",  # for English only
    'nl': 'GroNLP/mdebertav3-subjectivity-multilingual',
    'it': 'GroNLP/mdebertav3-subjectivity-multilingual',
}
_sentiment_models = {
    'en': "cardiffnlp/twitter-xlm-roberta-base-sentiment", # Ar, En, Fr, De, Hi, It, Sp, Pt
    'fr': "cardiffnlp/twitter-xlm-roberta-base-sentiment",
    'it': "cardiffnlp/twitter-xlm-roberta-base-sentiment",
    'nl': "nlptown/bert-base-multilingual-uncased-sentiment", # English, Dutch, German, French, Spanish, and Italian
}

registered_classifiers = {}


def main():

    args = parse_args()
    classifiers = [registered_classifiers[key] for key in args.classifiers]

    input_reader = [args.sentence] if args.sentence else sys.stdin

    csvwriter = csv.writer(sys.stdout)

    if args.header:
        csvwriter.writerow(args.classifiers)

    for s in input_reader:
        s = s.strip()
        csvwriter.writerow(classifier(args.lang)(s) for classifier in classifiers)


def parse_args():
    parser = argparse.ArgumentParser(description='Classify sentences along various dimensions.')
    parser.add_argument('sentence', nargs='?', type=str, default=None,
                        help='Sentence to process; otherwise read lines from stdin.')
    parser.add_argument('--lang', '--language', type=str, default='en', help='Language code (e.g., en, nl, fr, it)')
    parser.add_argument('--header', action='store_true', default=None,
                        help='Whether to print a csv-style header first.')
    for key, func in registered_classifiers.items():
        parser.add_argument(f'--{key}', action='append_const', dest='classifiers', const=key, help=func.__doc__)

    # TODO: If language not provided, auto-detect, maybe using fastlang or langdetect https://github.com/Mimino666/langdetect

    args = parser.parse_args()

    args.classifiers = ((seen := set())         # deduplicating a list maintaining order :D
                        or [c for c in args.classifiers if not (c in seen or seen.add(c))] if args.classifiers else registered_classifiers.keys())

    return args


def classifier_factory(name):
    """
    Decorator that registers classifiers.
    """
    def named_classifier(func):
        cached_func = functools.cache(func)
        registered_classifiers[name] = cached_func
        return cached_func
    return named_classifier


@classifier_factory('subj')
def load_subjectivity_classifier(language):

    model_name = _subjectivity_models[language]

    model = pipeline(
        task="text-classification",
        model=model_name,
        tokenizer=model_name,
        top_k=None,
    )

    index = ['LABEL_0', 'LABEL_1'].index    # TODO: Model-dependent...

    def subjectivity_classifier(text):
        result = model(text)[0]
        result = sorted(result, key=lambda x: index(x['label']))
        score = result[1]['score']
        return score

    return subjectivity_classifier


@classifier_factory('conc')
def load_concreteness_classifier(language):
    """
    To output concreteness scores.
    """

    # TODO: Update with MWE? https://osf.io/preprints/psyarxiv/m397u

    assert language == 'en'

    concreteness_ratings = {}
    with resources.files('auxiliary').joinpath('Concreteness_ratings_Brysbaert_et_al_BRM.tsv').open('r') as file:
        reader = csv.reader(file, delimiter="\t")
        next(reader)    # skip columns
        for row in reader:
            concreteness_ratings[row[0]] = float(row[2])

    word_re = re.compile(r'\b\w+\b')

    def concreteness_classifier(text):
        # TODO: Use lemmatization, getting rid of stop words etc.
        word_ratings = list(filter(None, (concreteness_ratings.get(w, None) for w in word_re.findall(text.lower()))))
        if not word_ratings:
            return None
        return (sum(word_ratings) / len(word_ratings) - 1) / 5

    return concreteness_classifier


@classifier_factory('sent')
def load_sentiment_classifier(language):

    model_name = _sentiment_models[language]
    model = pipeline("sentiment-analysis", model=model_name, tokenizer=model_name, top_k=None)

    labels = ['negative', 'neutral', 'positive']
    if language == 'nl':
        labels = ['1 star', '2 stars', '3 stars', '4 stars', '5 stars'] # TODO: Model-dependent...
    values = (-1 + (n * 2/(len(labels)-1)) for n in range(len(labels)))

    def sentiment_classifier(text):
        result = model(text)[0]
        probabilities = [r['score'] for r in sorted(result, key=lambda x: labels.index(x['label']))]
        aggregate = sum(value * prob for prob, value in zip(probabilities, values))
        return aggregate

    return sentiment_classifier


if __name__ == '__main__':
    main()
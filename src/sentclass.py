import sys
import argparse

import csv
import re

from importlib import resources


"""
Classif sentences along various dimensions.

Currently computes subjectivity based on a transformer model; concreteness simply by averaging word scores from a lookup table (Brysbaert et al.) 

Example:
$ sentclass "This is a sentence" --subj --conc
"""

def main():

    parser = argparse.ArgumentParser(description='Classify sentences along various dimensions.')
    parser.add_argument('sentence', nargs='?', type=str, default=None, help='Sentence to process; otherwise read lines from stdin.')
    parser.add_argument('--subj', '--subjectivity', action='store_true', help='Whether to output subjectivity scores.')
    parser.add_argument('--conc', '--concreteness', action='store_true', help='Whether to output concreteness scores.')

    classifier_map = {'subj': make_subjectivity_classifier,
                      'conc': make_concreteness_classifier}

    args = parser.parse_args(namespace=OrderedNamespace())

    lines = [args.sentence] if args.sentence else sys.stdin

    classifiers_requested = [key for key, val in args.ordered() if val and key in classifier_map] or classifier_map.keys()
    classifier_functions = [classifier_map[key]() for key in classifiers_requested]
    csvwriter = csv.writer(sys.stdout)

    for line in lines:
        line = line.strip()
        csvwriter.writerow(classifier(line) for classifier in classifier_functions)


class OrderedNamespace(argparse.Namespace):
    """
    From https://stackoverflow.com/a/59002780
    """
    def __init__(self, **kwargs):
        self.__dict__["_order"] = []
        super().__init__(**kwargs)
    def __setattr__(self, attr, value):
        super().__setattr__(attr, value)
        if attr in self._order:
            self.__dict__["_order"].clear()
        self.__dict__["_order"].append(attr)
    def ordered(self):
        return ((attr, getattr(self, attr)) for attr in self._order)


def make_subjectivity_classifier():

    from transformers import pipeline

    model = pipeline(
        task="text-classification",
        model="cffl/bert-base-styleclassification-subjective-neutral",
        top_k=None,
    )

    def classify_subjectivity(text):
        return model(text)[0][0]['score']

    return classify_subjectivity


def make_concreteness_classifier():

    # TODO: Update with MWE? https://osf.io/preprints/psyarxiv/m397u

    concreteness_ratings = {}
    with resources.files('auxiliary').joinpath('Concreteness_ratings_Brysbaert_et_al_BRM.tsv').open('r') as file:
        reader = csv.reader(file, delimiter="\t")
        next(reader)    # skip columns
        for row in reader:
            concreteness_ratings[row[0]] = float(row[2])

    word_re = re.compile(r'\b\w+\b')

    def classify_concreteness(text):
        word_ratings = [concreteness_ratings.get(w, None) for w in word_re.findall(text.lower())]
        if not word_ratings:
            return None
        return sum(filter(lambda x: x is not None, word_ratings)) / len(word_ratings)

    return classify_concreteness


if __name__ == '__main__':
    main()
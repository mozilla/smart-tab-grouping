"""Reproduce Firefox's topic-model input and c-TF-IDF keyword extraction.

Keep this module dependency-free. It mirrors CountVectorizer, CTfIdf, and
SmartTabGrouping.createModelInput in mozilla-central so dataset generation can
be tested without a Firefox build or the training environment.
"""

from __future__ import annotations

import math
import re
from collections.abc import Sequence


# toolkit/components/ml/content/nlp/StopWords.sys.mjs. This is scikit-learn's
# English list with Firefox's additions, "google" and "search".
ENGLISH_STOP_WORDS = frozenset(
    """
    a about above across after afterwards again against all almost alone along
    already also although always am among amongst amoungst amount an and another
    any anyhow anyone anything anyway anywhere are around as at back be became
    because become becomes becoming been before beforehand behind being below
    beside besides between beyond bill both bottom but by call can cannot cant co
    con could couldnt cry de describe detail do done down due during each eg eight
    either eleven else elsewhere empty enough etc even ever every everyone
    everything everywhere except few fifteen fifty fill find fire first five for
    former formerly forty found four from front full further get give go google
    had has hasnt have he hence her here hereafter hereby herein hereupon hers
    herself him himself his how however hundred i ie if in inc indeed interest
    into is it its itself keep last latter latterly least less ltd made many may
    me meanwhile might mill mine more moreover most mostly move much must my
    myself name namely neither never nevertheless next nine no nobody none noone
    nor not nothing now nowhere of off often on once one only onto or other others
    otherwise our ours ourselves out over own part per perhaps please put rather
    re same search see seem seemed seeming seems serious several she should show
    side since sincere six sixty so some somehow someone something sometime
    sometimes somewhere still such system take ten than that the their them
    themselves then thence there thereafter thereby therefore therein thereupon
    these they thick thin third this those though three through throughout thru
    thus to together too top toward towards twelve twenty two un under until up
    upon us very via was we well were what whatever when whence whenever where
    whereafter whereas whereby wherein whereupon wherever whether which while
    whither who whoever whole whom whose why will with within without would yet
    you your yours yourself yourselves
    """.split()
)

MAX_WORDS_PER_DOCUMENT = 3
MIN_THRESHOLD_FOR_KEYWORD = 0.05


def tokenize(document: str) -> list[str]:
    r"""Tokenize like JavaScript's ``replace(/[0-9]/g).split(/\W+/)``.

    JavaScript ``\w`` is ASCII-only, unlike Python's default regular-expression
    behavior, so the split expression is explicit here.
    """

    without_numbers = re.sub(r"[0-9]", "", document)
    return [
        token.lower()
        for token in re.split(r"[^A-Za-z0-9_]+", without_numbers)
        if token.lower() not in ENGLISH_STOP_WORDS
    ]


def extract_keywords(
    corpus: Sequence[str],
    max_words: int = MAX_WORDS_PER_DOCUMENT,
) -> list[list[str]]:
    """Return c-TF-IDF keywords for every corpus row, matching Firefox."""

    if not corpus:
        raise ValueError("corpus must not be empty")

    tokenized = [tokenize(document) for document in corpus]
    vocabulary: dict[str, int] = {}
    for document in tokenized:
        for token in document:
            if token and token not in vocabulary:
                vocabulary[token] = len(vocabulary)

    if not vocabulary:
        raise ValueError("corpus must contain at least one non-stopword token")

    counts: list[list[int]] = []
    for document in tokenized:
        row = [0] * len(vocabulary)
        for token in document:
            if token in vocabulary:
                row[vocabulary[token]] += 1
        counts.append(row)

    frequencies = [sum(row[index] for row in counts) for index in range(len(vocabulary))]
    average_samples = sum(sum(row) for row in counts) // len(counts)
    idf = [math.log(average_samples / frequency + 1) for frequency in frequencies]
    words = list(vocabulary)

    result: list[list[str]] = []
    for row in counts:
        row_sum = sum(abs(value) for value in row)
        scores = [
            (value / row_sum) * idf[index] if row_sum else 0.0
            for index, value in enumerate(row)
        ]
        # Python's sort, like modern JavaScript Array.sort, is stable. Vocabulary
        # insertion order therefore resolves equal-score ties in the same way.
        indices = sorted(range(len(scores)), key=lambda index: scores[index], reverse=True)
        selected: list[str] = []
        for index in indices:
            if scores[index] < MIN_THRESHOLD_FOR_KEYWORD or len(selected) >= max_words:
                break
            selected.append(words[index])
        result.append(selected)
    return result


def extract_group_keywords(
    documents: Sequence[str],
    other_documents: Sequence[str] = (),
) -> list[str]:
    """Extract the keywords Firefox uses when labeling one proposed group."""

    representative_documents = list(documents[:10])
    if len(representative_documents) <= 1:
        return []

    joined_documents = " ".join(representative_documents[:3])
    joined_other_documents = " ".join(other_documents)
    try:
        return extract_keywords([joined_documents, joined_other_documents])[0]
    except ValueError:
        # Firefox's implementation throws for an all-stopword corpus. Dataset
        # creation keeps that otherwise valid example with the no-keyword prompt.
        return []


def create_model_input(keywords: Sequence[str], documents: Sequence[str]) -> str:
    """Create exactly the string passed to Firefox's text2text pipeline."""

    joined_documents = " \n".join(documents[:10])
    if not keywords:
        return f"Topic from keywords: titles: \n{joined_documents}"
    return (
        f"Topic from keywords: {', '.join(keywords)}. titles: \n"
        f"{joined_documents}"
    )

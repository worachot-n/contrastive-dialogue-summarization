from nltk.corpus import wordnet
from nltk.corpus import stopwords
import nltk

nltk.download("punkt")
nltk.download("averaged_perceptron_tagger")
nltk.download("wordnet")
nltk.download("stopwords")


lemmatizer = nltk.stem.WordNetLemmatizer()  # Initiate nltk lemmatizer


def simple_tokenize(sentence):
    """Simple function for tokenizing text with nltk"""
    return nltk.word_tokenize(sentence)


def nltk_to_pos(pos):
    """Simple function for converting nltk pos to wordnet pos"""
    if pos.startswith("J"):
        return wordnet.ADJ
    elif pos.startswith("V"):
        return wordnet.VERB
    elif pos.startswith("N"):
        return wordnet.NOUN
    elif pos.startswith("R"):
        return wordnet.ADV
    else:
        return None


def lemmatize_text(text):
    """Function to lemmatize text according to the wordnet POS of each token"""

    tokenized_text = nltk.word_tokenize(text)
    POS_assigned_text = nltk.pos_tag(tokenized_text)

    available_POS = map(lambda x: (x[0], nltk_to_pos(x[1])), POS_assigned_text)

    lemmatized_text = [
        token if pos is None else lemmatizer.lemmatize(token, pos)
        for token, pos in available_POS
    ]

    return lemmatized_text


def build_tagger(original_tokens, lemmatized_tokens, topic_list, idx):
    tagged_tokens = []
    # Extract all the seed words according to the corresponding topic
    token_topics = topic_list
    original_list = original_tokens[idx]

    for j, token in enumerate(lemmatized_tokens[idx]):
        # If the lemmatized form of the token is in topic seeds, tag the original token
        if token.lower() in token_topics:
            # print(token.lower())
            if token.lower() not in stopwords.words("english"):
                # print("="*100)
                # print(token.lower())
                original_list[j] = "<t>" + original_list[j] + "</t>"

    tagged_tokens.append(" ".join(original_list))
    return tagged_tokens

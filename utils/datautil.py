import re

from sklearn.model_selection import train_test_split


def split(data, test_split=1500):
    return train_test_split(data, test_size=test_split, random_state=1)


def clean(data):
    x = []
    y = []

    for entry in data:
        # The entry is a Review dict
        # e.g.
        # {
        #   'index': '9336', 
        #   'title': 'Nu merita nici jumatate din pret!!', 
        #   'content': 'se misca execrabil!!! se blocheaza des!!! nici daca ar fi gratuita nu as mai luat-o!! bateria tine ok, dar la ce fost daca se blocheaza cand deschizi 2 aplicatii, doar atat?', 
        #   'starRating': '1'
        # }

        # Convert star rating into positive(1)/negative(0) label
        label = 0 if entry['starRating'] == '1' or entry['starRating'] == '1' else 1

        # Merge title and content into 'text'
        text = entry['title'] + ' ' + entry['content']

        # Remove non-alphanumeric
        text = ''.join(c for c in text if c.isalnum() or c == ' ')

        # Remove unnecessary whitespace
        text = re.sub('\s+', ' ', text).strip()

        # To lowercase
        text = text.lower()

        x.append(text)
        y.append(label)

    return x, y

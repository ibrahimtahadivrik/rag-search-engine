import json


def load_movies(location) -> list:
    movies = json.load(open(location))
    return movies
import gensim.downloader as api

def model(dimension):
    if dimension == 50:
        return api.load('glove-wiki-gigaword-50')
    elif dimension == 100:
        return api.load('glove-wiki-gigaword-100')
    elif dimension == 200:
        return api.load('glove-wiki-gigaword-200')
    elif dimension == 300:
        return api.load('glove-wiki-gigaword-300')
    else:
        raise ValueError('Invalid dimension: {}'.format(dimension))
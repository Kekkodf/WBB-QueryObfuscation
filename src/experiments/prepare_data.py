def get_vocab_embeddings(data, model):
    tokenized = data['query'].apply(lambda x: x.split())

    tokenized_list = [x for x in tokenized]

    vocab = set()
    for sentence in tokenized_list:
        for word in sentence:
            vocab.add(word)
    len_vocab=len(vocab)
    vocab_embeddings = {}
    for word in vocab:
        try:
            vocab_embeddings[word] = model[word]
        except:
            continue

    len_embs = len(vocab_embeddings)
    print('Lost {} words. Not in the vocabulary.'.format(len_vocab-len_embs))
    return vocab_embeddings
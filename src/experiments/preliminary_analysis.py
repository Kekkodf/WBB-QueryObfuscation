import glove
import queries
import distance
import matplotlib.pyplot as plt
import seaborn as sns
import obfuscator
sns.set_style("whitegrid")
sns.set_context("paper")
sns.set(font_scale=1.5)


model = glove.model(dimension=50)

data = queries.get_data()

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

word = 'tumor'

smoke_angle = []
#get angles
angles = []
for wrd in vocab_embeddings:
    if wrd == word:
        continue
    else:
        angle = distance.get_distance(word, wrd, model, 'cosine')
        angles.append((wrd, angle))
        smoke_angle.append((wrd, angle))

angles = sorted(angles, key=lambda x: x[1])

#print('Most close (angle) to {}'.format(word))
#print(angles[:20])

#get euclidean distance
smoke_distance = []
distances = []
for wrd in vocab_embeddings:
    if wrd == word:
        continue
    else:
        dist = distance.get_distance(word, wrd, model, 'euclidean')
        distances.append((wrd, dist))
        smoke_distance.append((wrd, dist))

distances = sorted(distances, key=lambda x: x[1])

#print('Most close (distance) to {}'.format(word))
#print(distances[:20])

#plot the words
words_angles = [w[0] for w in angles[:20]]
angles = [w[1] for w in angles[:20]]
words_distances = [w[0] for w in distances[:20]]
distances = [w[1] for w in distances[:20]]
            
smoke_word_angle = [w[0] for w in smoke_angle]
smoke_angle = [w[1] for w in smoke_angle]
smoke_word_distance = [w[0] for w in smoke_distance]
smoke_distance = [w[1] for w in smoke_distance]

smoke_data = []
for wrd in smoke_word_angle:
    for wrd2 in smoke_word_distance:
        try:
            smoke_data.append((wrd, smoke_angle[smoke_word_angle.index(wrd)], smoke_distance[smoke_word_distance.index(wrd)]))
            
        except:
            smoke_data.append((wrd2, distance.get_distance(word, wrd2, model, 'cosine'), distance.get_distance(word, wrd2, model, 'euclidean')))

data = []
for wrd in words_angles:
    for wrd2 in words_distances:
        try:
            data.append((wrd, angles[words_angles.index(wrd)], distances[words_distances.index(wrd)]))
        except:
            data.append((wrd2, distance.get_distance(word, wrd2, model, 'cosine'), distance.get_distance(word, wrd2, model, 'euclidean')))

#keep only unique values
data = list(set(data))
smoke_data = list(set(smoke_data))

print('Data length: {}'.format(len(data)))
print('Smoke data length: {}'.format(len(smoke_data)))

#check if words in data are in smoke data
for wrd, angle, dist in data:
    if (wrd, angle, dist) in smoke_data:
        print('Word {} is in smoke data'.format(wrd))

#plot a polar plot with the angles an distances, word as label in legend
fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111, polar=True)
ax.set_thetamin(0)
ax.set_thetamax(180)
ax.set_rmax(max(distances)+3)

for wrd, angle, dist in data:
    ax.scatter(angle, dist, marker='s', s=100, label=wrd)

for s_wrd, s_angle, s_dist in smoke_data:
    ax.scatter(s_angle, s_dist, marker='o', s=50, color='grey', alpha=0.2)

plt.title('Word: %s'%word, fontsize=20)
#place thenlegend outside the plot, in two columns
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., ncol=2)
plt.show()
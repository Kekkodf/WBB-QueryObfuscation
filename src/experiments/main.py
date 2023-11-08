import glove
import queries
import distance
import matplotlib.pyplot as plt
import seaborn as sns
import obfuscator
import math
import prepare_data as prep
sns.set_style("whitegrid")
sns.set_context("paper")
sns.set(font_scale=1.5)


model = glove.model(dimension=50)

data = queries.get_data()

vocab_embeddings = prep.get_vocab_embeddings(data, model)

word = 'cancer'

#obfuscate
##compute safe boxes (distance and angle based)
safe_box_distance = obfuscator.get_safe_box(word, vocab_embeddings, model, 'distance', d=4.5)
safe_box_angle = obfuscator.get_safe_box(word, vocab_embeddings, model, 'angle', a=math.radians(45))
##compute boundary words (for the two safe boxes)
boundary_word_distance = obfuscator.get_boundary_word(safe_box_distance, 'distance')
boundary_word_angle = obfuscator.get_boundary_word(safe_box_angle, 'angle')
##select obfuscated word
obfuscated_word_distance = obfuscator.get_obfuscated_word(boundary_word_distance, vocab_embeddings, model, 'distance', topn=15)
obfuscated_word_angle = obfuscator.get_obfuscated_word(boundary_word_angle, vocab_embeddings, model, 'angle', topn=15)

##print info
print('Distance based safe box for {}: {}'.format(word, safe_box_distance))
print('Distance based boundary word for {}: {}'.format(word, boundary_word_distance))
print('Distance based obfuscated word starting from {}: {}'.format(boundary_word_distance, obfuscated_word_distance))
print('-------------------------------------')
print('Angle based safe box for {}: {}'.format(word, safe_box_angle))
print('Angle based boundary word for {}: {}'.format(word, boundary_word_angle))
print('Angle based obfuscated word starting from {}: {}'.format(boundary_word_angle, obfuscated_word_angle))
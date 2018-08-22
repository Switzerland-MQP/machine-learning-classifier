import nltk

from nltk import word_tokenize, sent_tokenize
import gensim
from gensim.models.word2vec import Word2Vec
from sklearn.manifold import TSNE
import pandas as pd
from bokeh.io import output_notebook
from bokeh.plotting import show, figure


from nltk.corpus import gutenberg
# Punctuation and Tokenizer module
nltk.download('punkt')

# The Gutenberg dataset. A set of 18 books we can used
# to train upon.
nltk.download('gutenberg')

# Due to lack of resources, I'm not working with the full Gutenberg 
# dataset (18 books). If you got a GPU, you can just omit the
# ‘fileids’ parameter and all 18 books will be loaded.

gberg_sents = gutenberg.sents(fileids=['bible-kjv.txt',
                                       'austen-emma.txt',
                                       'austen-persuasion.txt',
                                       'austen-sense.txt',
                                       'carroll-alice.txt'])

# size = 64, dimensions
# sg = 1, use Skip-Gram. If zero, it will use CBOW
# window = 10, context words (10 to the left and 10 to the right)
# min_count = 5, ignore words with frequency lower than that
# seed = 42, the answer to the universe, life and everything.
# workers = 2, number of worker threads.

model = Word2Vec(sentences=gberg_sents, size=64, sg=1,
                 window=10, min_count=5, seed=42,
                 workers=2)

# Shows the coordinates of the word ‘house’ in the vector space.
#  print(model['house'])
#  print(model.wv.most_similar('house'))
#  print(model.wv.most_similar('day'))
#  print(model.wv.most_similar('father'))

#  print(model.doesnt_match('mother father daughter house'.split()))


tsne = TSNE(n_components=2, n_iter=300)
X = model[model.wv.vocab]
X_2d = tsne.fit_transform(X)

coords_df = pd.DataFrame(X_2d, columns=['x', 'y'])
coords_df['token'] = model.wv.vocab.keys()
print(coords_df.head())

# Plot the graph.
coords_df.plot.scatter('x', 'y', figsize=(8, 8),
                       marker='o', s=10, alpha=0.2)
output_notebook()

subset_df = coords_df.sample(n=1000)
p = figure(plot_width=600, plot_height=600)
p.text(x=subset_df.x, y=subset_df.y,
       text=subset_df.token)
show(p)

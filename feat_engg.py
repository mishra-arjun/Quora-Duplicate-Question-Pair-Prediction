import numpy as np
import pandas as pd
import gc
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
from nltk.corpus import stopwords
import _pickle as cPickle
from fuzzywuzzy import fuzz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline
from scipy.sparse import hstack
from tqdm import tqdm
import gensim
from scipy.stats import skew, kurtosis
from scipy.spatial.distance import cosine, cityblock, jaccard, canberra, euclidean, minkowski, braycurtis
from nltk import word_tokenize
stop_words = stopwords.words('english')
import pyemd
pal = sns.color_palette()

df = pd.read_csv('train.csv')
df.head()

#initial data analysis
print('Total number of question pairs in data: {}'.format(len(df)))
print('Duplicate pairs: {}%'.format(round(df['is_duplicate'].mean()*100, 2)))

qids = pd.Series(df['qid1'].tolist() + df['qid2'].tolist())
print('Total number of questions in the data: {}'.format(len(np.unique(qids))))
print('Number of questions that appear more than once: {}'.format(np.sum(qids.value_counts() > 1)))

plt.figure(figsize=(12, 5))
plt.hist(qids.value_counts(), bins=50)
plt.yscale('log', nonposy='clip')
plt.title('Log-Histogram of question appearance counts')
plt.xlabel('Number of occurences of question')
plt.ylabel('Number of questions')
print()

#distribution of characters
qs1 = pd.Series(df['question1'].tolist()).astype(str)
qs2 = pd.Series(df['question2'].tolist()).astype(str)
qs = qs1 + qs2
dist_qs1 = qs1.apply(len)
dist_qs2 = qs2.apply(len)
plt.figure(figsize=(15, 10))
plt.hist(dist_qs1, bins=200, range=[0, 200], color=pal[2], normed=True, label='Question 1')
plt.hist(dist_qs2, bins=200, range=[0, 200], color=pal[1], normed=True, alpha=0.5, label='Question 2')
plt.title('Normalised histogram of character count in questions', fontsize=15)
plt.legend()
plt.xlabel('Number of characters', fontsize=15)
plt.ylabel('Probability', fontsize=15)

print('Mean of # of characters in Questions 1 {:.2f}'.format(dist_qs1.mean()))
print('Std Dev of # of characters in Questions 1 {:.2f}'.format(dist_qs1.std()))
print('Minimum of # of characters in Questions 1 {:.2f}'.format(dist_qs1.min()))
print('Maximum of # of characters in Questions 1 {:.2f}'.format(dist_qs1.max()))
print('Mean of # of characters in Questions 2 {:.2f}'.format(dist_qs2.mean()))
print('Std Dev of # of characters in Questions 2 {:.2f}'.format(dist_qs2.std()))
print('Minimum of # of characters in Questions 2 {:.2f}'.format(dist_qs2.min()))
print('Maximum of # of characters in Questions 2 {:.2f}'.format(dist_qs2.max()))

#distribution of words
dist_qs1 = qs1.apply(lambda x: len(x.split(' ')))
dist_qs2 = qs2.apply(lambda x: len(x.split(' ')))

plt.figure(figsize=(15, 10))
plt.hist(dist_qs1, bins=50, range=[0, 50], color=pal[2], normed=True, label='Question 1')
plt.hist(dist_qs2, bins=50, range=[0, 50], color=pal[1], normed=True, alpha=0.5, label='Question 2')
plt.title('Normalised histogram of character count in questions', fontsize=15)
plt.legend()
plt.xlabel('Number of characters', fontsize=15)
plt.ylabel('Probability', fontsize=15)

print('Mean of # of characters in Questions 1 {:.2f}'.format(dist_qs1.mean()))
print('Std Dev of # of characters in Questions 1 {:.2f}'.format(dist_qs1.std()))
print('Minimum of # of characters in Questions 1 {:.2f}'.format(dist_qs1.min()))
print('Maximum of # of characters in Questions 1 {:.2f}'.format(dist_qs1.max()))
print('Mean of # of characters in Questions 2 {:.2f}'.format(dist_qs2.mean()))
print('Std Dev of # of characters in Questions 2 {:.2f}'.format(dist_qs2.std()))
print('Minimum of # of characters in Questions 2 {:.2f}'.format(dist_qs2.min()))
print('Maximum of # of characters in Questions 2 {:.2f}'.format(dist_qs2.max()))

#semantics
qmarks = np.mean(qs.apply(lambda x: '?' in x))
math = np.mean(qs.apply(lambda x: '[math]' in x))
fullstop = np.mean(qs.apply(lambda x: '.' in x))
capital_first = np.mean(qs.apply(lambda x: x[0].isupper()))
capitals = np.mean(qs.apply(lambda x: max([y.isupper() for y in x])))
numbers = np.mean(qs.apply(lambda x: max([y.isdigit() for y in x])))

print('Questions with question marks: {:.2f}%'.format(qmarks * 100))
print('Questions with [math] tags: {:.2f}%'.format(math * 100))
print('Questions with full stops: {:.2f}%'.format(fullstop * 100))
print('Questions with capitalised first letters: {:.2f}%'.format(capital_first * 100))
print('Questions with capital letters: {:.2f}%'.format(capitals * 100))
print('Questions with numbers: {:.2f}%'.format(numbers * 100))

#word share ratio feature
#getting English stopwords
stops = set(stopwords.words("english"))

def word_match_share(row):
    q1words = {}
    q2words = {}
    for word in str(row['question1']).lower().split():
        if word not in stops:
            q1words[word] = 1
    for word in str(row['question2']).lower().split():
        if word not in stops:
            q2words[word] = 1
    if len(q1words) == 0 or len(q2words) == 0:
        return 0
    shared_words_in_q1 = [w for w in q1words.keys() if w in q2words]
    shared_words_in_q2 = [w for w in q2words.keys() if w in q1words]
    R = (len(shared_words_in_q1) + len(shared_words_in_q2))/(len(q1words) + len(q2words))
    return R

plt.figure(figsize=(15, 5))
word_match = df.apply(word_match_share, axis=1, raw=True)
plt.hist(word_match[df['is_duplicate'] == 0], bins=20, normed=True, label='Not Duplicate')
plt.hist(word_match[df['is_duplicate'] == 1], bins=20, normed=True, alpha=0.7, label='Duplicate')
plt.legend()
plt.title('Label distribution over Word Match ratio', fontsize=15)
plt.xlabel('Word Match ratio', fontsize=15)

#FEATURE ENGINEERING

#Basic features
# length of question 1 and question 2
df['len_q1'] = df.question1.apply(lambda x: len(str(x)))
df['len_q2'] = df.question2.apply(lambda x: len(str(x)))

#ratio of question lengths
df['len_ratio'] = df.len_q1 / df.len_q2

#ratio of difference in lengths of questions to total length of questions
df['diff_len_ratio'] = abs(df.len_q1 - df.len_q2) / (df.len_q1 + df.len_q2)

# Number of words in question1
df['len_word_q1'] = df.question1.apply(lambda x: len(str(x).split()))

# Number of words in question2
df['len_word_q2'] = df.question2.apply(lambda x: len(str(x).split()))

# ratio of number of words
df['len_word_ratio'] = df.len_word_q1 / df.len_word_q2

# ratio of difference in these lengths to total length
df['diff_len_word_ratio'] = abs(df.len_word_q1 - df.len_word_q2) / (df.len_word_q1 + df.len_word_q2)

# Number of common words in question1 and question2
df['common_words'] = df.apply(lambda x: len(set(str(x['question1']).lower().split()).intersection(set(str(x['question2']).lower().split()))), axis=1)

# ratio of number of common words to average length of the questions
df['common_words_ratio'] = 2* df.common_words / (df.len_word_q1 + df.len_word_q2)

df.to_csv('fs1.csv', index=False)

#Fuzzy features

# Q-ratio
df['fuzz_Qratio'] = df.apply(lambda x: fuzz.QRatio(str(x['question1']), str(x['question2'])), axis=1)

# W-ratio
df['fuzz_WRatio'] = df.apply(lambda x: fuzz.WRatio(str(x['question1']), str(x['question2'])), axis=1)

# Partial ratio
df['fuzz_partial_ratio'] = df.apply(lambda x: fuzz.partial_ratio(str(x['question1']), str(x['question2'])), axis=1)

# Partial token set ratio
df['fuzz_partial_token_set_ratio'] = df.apply(lambda x: fuzz.partial_token_set_ratio(str(x['question1']), str(x['question2'])), axis=1)

# Partial token sort ratio
df['fuzz_partial_token_sort_ratio'] = df.apply(lambda x: fuzz.partial_token_sort_ratio(str(x['question1']), str(x['question2'])), axis=1)

# Token set ratio
df['fuzz_token_set_ratio'] = df.apply(lambda x: fuzz.token_set_ratio(str(x['question1']), str(x['question2'])), axis=1)

# Token sort ratio
df['fuzz_token_sort_ratio'] = df.apply(lambda x: fuzz.token_sort_ratio(str(x['question1']), str(x['question2'])), axis=1)

df.to_csv('fs12.csv', index=False)

#TF-IDF and SVD

# creating corpuses on which to create the TF-IDF features
corpus1 = df.question1.apply(lambda x: str(x))
corpus2 = df.question2.apply(lambda x: str(x))
corpus3 = df.question1.apply(lambda x: str(x)) + df.question2.apply(lambda x: str(x))

# creating the TF-IDF and SVD transformers
tf = TfidfVectorizer(analyzer='word', ngram_range=(1,3), min_df = 0, stop_words = 'english')
svd = TruncatedSVD(n_components=100, n_iter=5, random_state=42, algorithm='randomized')

# creating a pipeline of TF-IDF and SVD transformation
svd_transformer = Pipeline([('tfidf', tf), ('svd', svd)])

# creating features according to (1) above
svd_matrix1 = svd_transformer.fit_transform(corpus1)
svd_matrix2 = svd_transformer.fit_transform(corpus2)
tfidf_feat1 = np.hstack((svd_matrix1, svd_matrix2))
cPickle.dump(tfidf_feat1, open('tfidf1.pkl', 'wb'), -1)

# creating features according to (2) above
tfidf_matrix = hstack((tf.fit_transform(corpus1), tf.fit_transform(corpus2)))
tfidf_feat2 = svd.fit_transform(tfidf_matrix)
cPickle.dump(tfidf_feat2, open('tfidf2.pkl', 'wb'), -1)

# creating features according to (3) above
tfidf_feat3 = svd_transformer.fit_transform(corpus3)
cPickle.dump(tfidf_feat3, open('tfidf3.pkl', 'wb'), -1)

# word2vec

# function to calculate sentence to vector
def sent2vec(s):
    words = str(s).lower()
    words = word_tokenize(words)
    words = [w for w in words if not w in stop_words]
    words = [w for w in words if w.isalpha()]
    M = []
    for w in words:
        try:
            M.append(model[w])
        except:
            continue
    M = np.array(M)
    v = M.sum(axis=0)
    return v / np.sqrt((v ** 2).sum())

model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)

# initializing vectors for questions 1 and 2
question1_vectors = np.zeros((df.shape[0], 300))
error_count = 0

for i, q in tqdm(enumerate(df.question1.values)):
    question1_vectors[i, :] = sent2vec(q)

question2_vectors  = np.zeros((df.shape[0], 300))
for i, q in tqdm(enumerate(df.question2.values)):
    question2_vectors[i, :] = sent2vec(q)

# cosine distance
df['cosine_distance'] = [cosine(x, y) for (x, y) in zip(np.nan_to_num(question1_vectors), np.nan_to_num(question2_vectors))]

# cityblock distance
df['cityblock_distance'] = [cityblock(x, y) for (x, y) in zip(np.nan_to_num(question1_vectors), np.nan_to_num(question2_vectors))]

# jaccard distance
df['jaccard_distance'] = [jaccard(x, y) for (x, y) in zip(np.nan_to_num(question1_vectors), np.nan_to_num(question2_vectors))]

# canberra distance
df['canberra_distance'] = [canberra(x, y) for (x, y) in zip(np.nan_to_num(question1_vectors), np.nan_to_num(question2_vectors))]

# euclidean distance
df['euclidean_distance'] = [euclidean(x, y) for (x, y) in zip(np.nan_to_num(question1_vectors), np.nan_to_num(question2_vectors))]

# minkowski distance
df['minkowski_distance'] = [minkowski(x, y, 3) for (x, y) in zip(np.nan_to_num(question1_vectors), np.nan_to_num(question2_vectors))]

# braycurtis distance
df['braycurtis_distance'] = [braycurtis(x, y) for (x, y) in zip(np.nan_to_num(question1_vectors), np.nan_to_num(question2_vectors))]

# skew of vector for Question 1
df['skew_q1vec'] = [skew(x) for x in np.nan_to_num(question1_vectors)]

# skew of vector for question 2
df['skew_q2vec'] = [skew(x) for x in np.nan_to_num(question2_vectors)]

# Kurtosis of vector for Question 1
df['kur_q1vec'] = [kurtosis(x) for x in np.nan_to_num(question1_vectors)]

# Kurtosis of vector for Question 2
df['kur_q2vec'] = [kurtosis(x) for x in np.nan_to_num(question2_vectors)]

#dumping question1 and question2 word2vec vectors
cPickle.dump(question1_vectors, open('q1_w2v.pkl', 'wb'), -1)
cPickle.dump(question2_vectors, open('q2_w2v.pkl', 'wb'), -1)

df.to_csv('fs124.csv', index=False)
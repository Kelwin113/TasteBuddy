import nltk
import pandas as pd
from surprise import Dataset, Reader, SVD
from sklearn.metrics.pairwise import linear_kernel
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
from prettytable import PrettyTable

# NLTK
# nltk.download('punkt')

# Pandas
pd.DataFrame()

# Surprise
reader = Reader()
data = Dataset.load_from_df(pd.DataFrame({'uid': [1, 2, 3], 'iid': [4, 5, 6], 'rating': [7, 8, 9]}), reader)
algo = SVD()
trainset = data.build_full_trainset()
algo.fit(trainset)

# scikit-learn
linear_kernel([[1, 2], [3, 4]], [[5, 6], [7, 8]])

# NLTK
word_tokenize('Hello world')

# PrettyTable
x = PrettyTable()
x.field_names = ["City name", "Area", "Population", "Annual Rainfall"]
x.add_row(["Adelaide", 1295, 1158259, 600.5])
print(x)

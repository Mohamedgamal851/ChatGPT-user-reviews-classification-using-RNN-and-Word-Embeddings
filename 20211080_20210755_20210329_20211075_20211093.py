import pandas as pd
import re
import nltk
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, LSTM, Dense, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.utils import class_weight
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings("ignore")


nltk.download('stopwords')
nltk.download('punkt')


df = pd.read_csv('chatgpt_reviews.csv').dropna()
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = word_tokenize(text.lower())
    return ' '.join([word for word in tokens if word not in stop_words])

df['Review'] = df['Review'].apply(clean_text)
df['label'] = df['label'].map({'POSITIVE': 1, 'NEGATIVE': 0})


embedding_index = {}
with open("glove.6B.100d.txt", encoding='utf-8') as f:
    for line in f:
        values = line.split()
        word = values[0]
        vector = np.asarray(values[1:], dtype='float32')
        embedding_index[word] = vector

print(f"Loaded {len(embedding_index)} GloVe vectors.")

# Parameters
MAX_WORDS = 10000
embedding_dim = 100
# split_ratios = [0.2, 0.3]
# padding_lengths = [100, 150, 200]


results = []
split = 0.2
max_len = 100

# for split in split_ratios:
#     for max_len in padding_lengths:
       
tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token="<OOV>")
tokenizer.fit_on_texts(df['Review'])
sequences = tokenizer.texts_to_sequences(df['Review'])
padded = pad_sequences(sequences, maxlen=max_len, padding='post', truncating='post')

                
X_train, X_test, y_train, y_test = train_test_split(
    padded, df['label'], test_size=split, random_state=42, stratify=df['label']
)

                
word_index = tokenizer.word_index
embedding_matrix = np.zeros((MAX_WORDS, embedding_dim))
for word, i in word_index.items():
    if i < MAX_WORDS:
        vec = embedding_index.get(word)
        if vec is not None:
            embedding_matrix[i] = vec
oov_index = tokenizer.word_index.get('<OOV>')
if oov_index and oov_index < MAX_WORDS:
    embedding_matrix[oov_index] = np.random.normal(scale=0.6, size=(embedding_dim,))

                
weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weights_dict = {0: weights[0], 1: weights[1]}

def build_model(type='rnn'):
            model = Sequential()
            model.add(Embedding(input_dim = MAX_WORDS, output_dim = embedding_dim, weights=[embedding_matrix],
                                        input_length=max_len, trainable=True))
            if type == 'rnn':
                model.add(Bidirectional(SimpleRNN(64)))
                model.build(input_shape=(None, max_len))
                model.summary()
            else:
                model.add(Bidirectional(LSTM(64, dropout=0.3, recurrent_dropout=0.3)))
                model.build(input_shape=(None, max_len))
                model.summary()
            model.add(Dense(1, activation='sigmoid'))
            model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
            return model

for model_type in ['rnn', 'lstm']:
            model = build_model(model_type)
            early_stop = EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)

            history = model.fit(X_train, y_train, epochs=10, batch_size=32,
                                validation_split=0.2, class_weight=class_weights_dict,
                                callbacks=[early_stop], verbose=0)

            y_pred = (model.predict(X_test) > 0.5).astype("int32")
            acc = accuracy_score(y_test, y_pred)
            results.append({
                'Split Ratio': 1 - split,
                'Padding Length': max_len,
                'Model': 'SimpleRNN' if model_type == 'rnn' else 'LSTM',
                'Accuracy': round(acc, 4)
            })
            print(f"Finished: Split={1 - split}, MaxLen={max_len}, Model={model_type.upper()}, Accuracy={acc:.4f}")


results_df = pd.DataFrame(results)
print("\n🔍 Accuracy Comparison Table:\n")
print(results_df.pivot_table(index=['Split Ratio', 'Padding Length'], columns='Model', values='Accuracy'))

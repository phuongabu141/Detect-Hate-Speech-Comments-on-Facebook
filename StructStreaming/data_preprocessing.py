import re
import pandas as pd
import numpy as np
from pyvi.ViTokenizer import ViTokenizer
from tensorflow.keras.preprocessing import text, sequence
from tensorflow.keras.utils import to_categorical
from pyspark.sql import types as t
from pyspark.sql import functions as f
import pickle

sequence_length = 100

with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

STOPWORDS = 'vietnamese-stopwords-dash.txt'
with open(STOPWORDS, "r") as ins:
    stopwords = []
    for line in ins:
        dd = line.strip('\n').strip('\"')
        stopwords.append(dd)
    stopwords = set(stopwords)


def filter_stop_words(train_sentences, stop_words):
    new_sent = [word for word in train_sentences.split()
                if word not in stop_words]
    train_sentences = ' '.join(new_sent)

    return train_sentences


def deEmojify(text):
    regrex_pattern = re.compile(pattern="["
                                u"\U0001F600-\U0001F64F"  # emoticons
                                u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                                u"\U0001F680-\U0001F6FF"  # transport & map symbols
                                u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                                "]+", flags=re.UNICODE)
    return regrex_pattern.sub(r'', text)


def preprocess(text, tokenized=True, lowercased=True):
    text = ViTokenizer.tokenize(text) if tokenized else text
    text = filter_stop_words(text, stopwords)
    text = deEmojify(text)
    text = text.lower() if lowercased else text
    return str(text)


def make_featues_text(X):
    text = str(X).strip().replace('"', '')
    X = tokenizer.texts_to_sequences([X])
    X = sequence.pad_sequences(X, maxlen=sequence_length)
    return X[0].tolist()


preprocess_udf = f.udf(preprocess, t.StringType())
make_featues_text_udf = f.udf(make_featues_text, t.ArrayType(t.IntegerType()))
make_label_udf = f.udf(lambda v: to_categorical(v, num_classes=3).tolist(), t.ArrayType(t.DoubleType()))

user_regex = r"(@\w{1,15})"
hashtag_replace_regex = "#(\w{1,})"
url_regex = r"((https?|ftp|file):\/{2,3})+([-\w+&@#/%=~|$?!:,.]*)|(www.)+([-\w+&@#/%=~|$?!:,.]*)"
email_regex = r"[\w.-]+@[\w.-]+\.[a-zA-Z]{1,}"


def pre_process_data(data):
    # Loại bỏ @Mention khỏi comment
    data = (data.withColumn("comment", f.regexp_replace(f.col("comment"), user_regex, ""))
            # Loại bỏ #Hashtag khỏi text
            .withColumn("comment", f.regexp_replace(f.col("comment"), hashtag_replace_regex, "$1"))
            # Loại bỏ URL khỏi text
            .withColumn("comment", f.regexp_replace(f.col("comment"), url_regex, ""))
            # Loại bỏ Email khỏi text
            .withColumn("comment", f.regexp_replace(f.col("comment"), email_regex, ""))
            # Loại bỏ các khoảng trắng thừa trong câu
            .withColumn("comment", f.regexp_replace(f.col("comment"), " +", " "))
            # Loại vỏ các khoảng trắng đầu và cuối câu
            .withColumn("comment", f.trim(f.col("comment")))
            # Chuẩn hoá viết thường
            .withColumn("comment", f.lower(f.col("comment")))
            # Giữ lại các dòng mà đoạn text có nội dung
            .filter(f.col("comment") != "")
            # Tiền xử lý dữ liệu
            .withColumn('comment_clean', preprocess_udf('comment'))
            # Encode text
            .withColumn('comment_encoded', make_featues_text_udf('comment_clean')))

    return data

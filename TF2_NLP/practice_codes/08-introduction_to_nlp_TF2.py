# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 11:34:55 2025

@author: ismt
"""

#NLP problems - seq2seq
from helper_functions import unzip_data, plot_loss_curves, compare_historys

#get text dataset
#the dataset we're going to use is Kaggles ıntroduction to NLP dataset
#text disaster or not disaster

#unzip data
#unzip_data("nlp_getting_started.zip")

#become one with the data
#visualize dataaaaaaaaaa

import pandas as pd

train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")
train_df.head()

train_df["text"][0]

#shuffle train dataframe
train_df_shuffled = train_df.sample(frac=1, random_state=42)
train_df_shuffled.head()

#look test dataframe
test_df.head()

#how many exps of each class
train_df.target.value_counts()

#how many total samples?
len(train_df), len(test_df)

#visualize some random training examples

import random
random_idx = random.randint(0, len(train_df)-5)
for row in train_df_shuffled[["text", "target"]][random_idx:random_idx+5].itertuples():
    _, text, target = row
    print(f"Target: {target}", "real disaster" if target>0 else "(not real disaster)")
    print(f"Text:\n{text}\n")
    
#split data.. train-validation 

from sklearn.model_selection import train_test_split
train_sentences, val_sentences, train_labels, val_labels = train_test_split(train_df_shuffled["text"].to_numpy(),
                                                                            train_df_shuffled["target"].to_numpy(),
                                                                            test_size=0.1, #use %10 of train data for validation_split
                                                                            random_state=42
                                                                            )

len(train_sentences), len(train_labels), len(val_sentences), len(val_labels)

#check the first 10 samples
train_sentences[:10], train_labels[:10]
##convert text into numbers.

#when dealing with text problem and one of the first thing we have to do 
#convert text to numbers.. tokenization, embedding
##text vectorization (tokenization)

import tensorflow as tf
from tensorflow.keras.layers import TextVectorization

#use default textvectorization parameters
text_vectorizer = TextVectorization(max_tokens=20_000, #how many words in the vocabulary
                                    standardize="lower_and_strip_punctuation",
                                    split="whitespace",
                                    ngrams=None, #create groups of N word
                                    output_mode="int",
                                    output_sequence_length=None, #how long do you want your sequences to be
                                    pad_to_max_tokens=True)                                                                   


#find the average number of tokens(words) in traini tweets
round(sum([len(i.split()) for i in train_sentences]) / len(train_sentences))

#setup text vectorization variables
max_vocab_length=10_000
max_length=15 #max len our sequences will be
text_vectorizer = TextVectorization(max_tokens=max_vocab_length,
                                    output_mode="int",
                                    output_sequence_length=max_length)

#fit the text vectorizer to train text
text_vectorizer.adapt(train_sentences) #not test or validation data..

#create sample sentences and tokenize it
sample_sentence = "There's a flood in my street"
text_vectorizer([sample_sentence])

#choose random sentence from train dataset and tokenize
random_sentence = random.choice(train_sentences)
print(f"original text : \n {random_sentence}\
      \n\nVectorized version:")
text_vectorizer([random_sentence])

#get unique words in vocabulary

words_in_vocab = text_vectorizer.get_vocabulary() #get all of the unique words in train data
top_5_words = words_in_vocab[:5]
bottom_5_words = words_in_vocab[-5:]

#create embedding..
#turns positive integers(indexes) into dense vectors of fixed size
#To make use tensorflow
from tensorflow.keras.layers import Embedding
#the parameters we care most about 
#input_dim : the size of our vocab
#output_dim : the size of output embedding vector
#input_length : length of sequences being passed to embedding layer
embedding = Embedding(input_dim=max_vocab_length,
                      output_dim=128,
                      input_length=max_length)

#get a random sentence 
random_sentence = random.choice(train_sentences)
print(f"original text : \n {random_sentence}\
      \n\nEmbedded version:")
sample_embed = embedding(text_vectorizer([random_sentence]))
print(sample_embed)

#check sigle token embed
sample_embed[0][0].shape, random_sentence[0]

#we will start baseline and move..
#model 0 : naive bayes (baseline)
#model 1: FFNN
#model 2: LSTM RNN
#model 3: GRU RNN
#model 4: Bidirectional LSTM
#model 5: 1D conv model
#model 6: pretrained feature extractor transfer learninng
#model 7: same as model 6 but %10 train data.

#Create model.. build model.. fit model.. evaulate model.. its classic path
#scikit learn, naive bayes text classifier with tfidf
#BASELINE MODEL
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

model_0 = Pipeline([
    ("tfidf", TfidfVectorizer()), #converts word to numbers
    ("clf", MultinomialNB()) # model
    ])

#fit pipeline to data
model_0.fit(train_sentences, train_labels)

#evaulate baseline model
baseline_score = model_0.score(val_sentences, val_labels) #like tf evaluate 
print(f"Baseline model achieves accuracy: {baseline_score*100:.2f}%")

baseline_preds = model_0.predict(val_sentences)
baseline_preds[:20]

#CREATE evaluation function.. y_true, y_pred
#we could evaluate with different metrics accuracy, precision, recall, f1-score
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
def calculate_results(y_true, y_pred):
    """
    calculates model accuracy, precision, f1score binary classificatio

    """
    model_accuracy = accuracy_score(y_true, y_pred)*100
    model_precision, model_recall, model_f1, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted")
    model_results = {
        "accuracy": model_accuracy,
        "precision": model_precision,
        "recall": model_recall,
        "f1": model_f1
        }
    
    return model_results

baseline_results = calculate_results(y_true=val_labels,
                                     y_pred=baseline_preds)
    
print(baseline_results)

#MODEL 1 FeedForward NN for text data
inputs = tf.keras.layers.Input(shape=(1,), dtype=tf.string) #one sequence of data
x = text_vectorizer(inputs) #turn text to numbers
x = embedding(x) #create embedding
x = tf.keras.layers.GlobalMaxPool1D()(x) #we can use also avarage pool
outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
model_1 = tf.keras.models.Model(inputs, outputs, name="model_1_dense")

model_1.compile(loss="binary_crossentropy",
                optimizer=tf.keras.optimizers.Adam(),
                metrics=["accuracy"])

model_1_hist = model_1.fit(x=train_sentences, 
                           y=train_labels,
                           validation_data=(val_sentences, val_labels),
                           epochs=5)

model_1.evaluate(val_sentences, val_labels)
model_1_pred_probs = model_1.predict(val_sentences)
#convert probabilities to label
model_1_preds = tf.squeeze(tf.round(model_1_pred_probs))
model_1_preds[:10]

#calculate results
model_1_results = calculate_results(y_true=val_labels,
                                   y_pred=model_1_preds)

model_1_results

import numpy as np
np.array(list(model_1_results.values())) > np.array(list(baseline_results.values()))

#whats happening in embedding layer? deatiled inspection of embedding layer
#visualizing embedding
#first get vocabulary from text vectorization layer
words_in_vocab = text_vectorizer.get_vocabulary()
len(words_in_vocab), words_in_vocab[:5]

#get the weight matrix of embed layer
embed_weights = model_1.get_layer("embedding").get_weights()[0]
embed_weights.shape #same size as vocab size and embedding_dim(V, D) V:all vocabs in our language

# #To do so, tensorflow has a handy tool
# #projector
# import io
# out_v = io.open('vectors.tsv', 'w', encoding='utf-8')
# out_m = io.open('metadata.tsv', 'w', encoding='utf-8')

# for index, word in enumerate(words_in_vocab):
#   if index == 0:
#     continue  # skip 0, it's padding.
#   vec = embed_weights[index]
#   out_v.write('\t'.join([str(x) for x in vec]) + "\n")
#   out_m.write(word + "\n")
# out_v.close()
# out_m.close()

#RNN's are useful for sequence data..
#model 2: LSTM

#text --> tokenize --> embedding --> Layer(rnn,cnn,dense) --> output(probability)
from tensorflow.keras import layers 
inputs = layers.Input(shape=(1,), dtype=tf.string)
x = text_vectorizer(inputs)
x = embedding(x)
print(x.shape)
x = layers.LSTM(64, return_sequences=True)(x) #when you stacking RNN celss together you need to return sequences=True
print(x.shape)
x = layers.LSTM(64)(x)
print(x.shape)
#x = layers.Dense(64, activation="relu")(x)
outputs = layers.Dense(1, activation='sigmoid')(x)
model_2 = tf.keras.models.Model(inputs, outputs, name='model_2_LSTM')
#LSTM layer expectin ndim=3 its very important #return sequences returns 3-d data.. so we need it stacking LSTM RNN units together
#(batch, timestep, feature)

#compile
model_2.compile(loss="binary_crossentropy",
                optimizer=tf.keras.optimizers.Adam(),
                metrics=["accuracy"])

model_2_hist = model_2.fit(train_sentences,
                           train_labels,
                           epochs=5,
                           validation_data=(val_sentences, val_labels))



model_2_pred_probs = model_2.predict(val_sentences)
model_2_pred_probs[:10]

model_2_preds = tf.squeeze(tf.round(model_2_pred_probs))
model_2_preds[:10]

model_2_results = calculate_results(y_true=val_labels, y_pred=model_2_preds)
print(model_2_results)

#MODEL 3 : GRU #similar LSTM, but has less parameter
from tensorflow.keras import layers
inputs = layers.Input(shape=(1,), dtype=tf.string)
x = text_vectorizer(inputs)
x = embedding(x)
x = layers.GRU(64)(x)
# x = layers.GRU(64, return_sequences=True)(x)
# x = layers.GlobalAveragePooling1D()(x)
#x = layers.LSTM(64)(x)

#x = layers.Dense(64, activation="relu")(x)
outputs = layers.Dense(1, activation="sigmoid")(x)
model_3 = tf.keras.models.Model(inputs, outputs, name="model_3_GRU")

model_3.compile(loss="binary_crossentropy",
                optimizer=tf.keras.optimizers.Adam(),
                metrics=["accuracy"])

model_3.fit(train_sentences,
            train_labels,
            epochs=5,
            validation_data=(val_sentences, val_labels))

model_3_pred_probs = model_3.predict(val_sentences)
model_3_pred_probs.shape

model_3_preds = tf.squeeze(tf.round(model_3_pred_probs))
model_3_preds[:10]

model_3_results = calculate_results(y_true=val_labels,
                                    y_pred = model_3_preds)
model_3_results

# MODEL 4 Bidirectional RNN Model
#normal RNN's goes from left to right just like reading english sentences 
#however bidirectional rNN goes form right to left as weel as left to right
#build bidirectionak rnn 
from tensorflow.keras import layers
inputs = layers.Input(shape=(1,), dtype=tf.string)
x = text_vectorizer(inputs)
x = embedding(x)
#x = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(x) #shape is 64 x 2 because goes both 2 ways 
x = layers.Bidirectional(layers.LSTM(64))(x)
outputs = layers.Dense(1, activation="sigmoid")(x)
model_4 = tf.keras.models.Model(inputs, outputs, name="model_4_bidirectional")

model_4.compile(loss="binary_crossentropy",
                optimizer=tf.keras.optimizers.Adam(),
                metrics=["accuracy"])

model_4.fit(train_sentences,
            train_labels,
            epochs=5,
            validation_data=(val_sentences, val_labels))

model_4_pred_probs = model_4.predict(val_sentences)
model_4_preds = tf.squeeze(tf.round(model_4_pred_probs))
model_4_preds[:5]

model_4_results = calculate_results(y_true=val_labels,
                                    y_pred=model_4_preds)

model_4_results

#Model 5 1D CNN Model..
#We used CNN's for images typcally 2D but text data is 1D
#we used Conv2D but now we will use Conv1D
#typical structure for 1 D text
#Inputs -- token -- embed -- Conv1D+pooling -- outputs 

#test embedding layer conv1dlayer and maxpooling
embedding_test = embedding(text_vectorizer(["this is a test sentence"]))
conv_1d = layers.Conv1D(filters=32,
                        kernel_size=5,
                        strides=1,#default
                        activation="relu",
                        padding="same")
conv_1d_output = conv_1d(embedding_test) #pass test embedding 
max_pool = layers.GlobalMaxPool1D()
max_pool_output = max_pool(conv_1d_output) #get the most important feature..

embedding_test.shape, conv_1d_output.shape, max_pool_output.shape

#create 1-d conv network
inputs = layers.Input(shape=(1,), dtype=tf.string)
x = text_vectorizer(inputs)
x = embedding(x)
x = layers.Conv1D(filters=64, kernel_size=5,
                  activation="relu",
                  padding="valid")(x)
x = layers.GlobalMaxPool1D()(x)
#x = layers.Dense(64, activation="relu")(x)
outputs = layers.Dense(1, activation="sigmoid")(x)
model_5 = tf.keras.models.Model(inputs, outputs, name="conv1d_model")

model_5.compile(loss="binary_crossentropy",
                optimizer=tf.keras.optimizers.Adam(),
                metrics=["accuracy"])

hist_5 = model_5.fit(train_sentences, train_labels,
            validation_data=(val_sentences, val_labels),
            epochs=5)


model_5_pred_probs = model_5.predict(val_sentences)
model_5_preds = tf.squeeze(tf.round(model_5_pred_probs))

model_5_results = calculate_results(y_true=val_labels,
                                    y_pred=model_5_preds)

model_5_results

#using tensorflow hub pretrained word embeddings (transfer learning for NLP)
#USE feature extractor
# Example of pretrained embedding with universal sentence encoder - https://tfhub.dev/google/universal-sentence-encoder/4
import tensorflow_hub as hub
embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4") # load Universal Sentence Encoder
embed_samples = embed([sample_sentence,
                      "When you call the universal sentence encoder on a sentence, it turns it into numbers."])

print(embed_samples[0][:50])
#create keras layer using pretrained layer 
sentence_encoder_layer = hub.KerasLayer("https://tfhub.dev/google/universal-sentence-encoder/4",
                                        input_shape=[],
                                        dtype=tf.string,
                                        trainable=False,
                                        name="USE")

#create model using sequential api
model_6 = tf.keras.Sequential([
    sentence_encoder_layer,
    layers.Dense(64, activation="relu"),
    layers.Dense(1, activation="sigmoid")
    ], name="model_6_USE")

model_6.compile(loss="binary_crossentropy",
                optimizer=tf.keras.optimizers.Adam(),
                metrics=["accuracy"])

hist_6 = model_6.fit(train_sentences,
                     train_labels,
                     epochs=5,
                     validation_data=(val_sentences, val_labels))

#predictions
model_6_pred_probs = model_6.predict(val_sentences)
model_6_preds = tf.squeeze(tf.round(model_6_pred_probs))
model_6_preds[:5]

model_6_results = calculate_results(y_true=val_labels,
                                    y_pred=model_6_preds)

#MODEL 7 same as model 6 but %10 percent of training data
# #
# train_10_percent = train_df_shuffled[["text", "target"]].sample(frac=0.1, random_state=42)
# train_10_percent.head(), len(train_10_percent), len(train_df_shuffled)
# train_sentences_10_percent = train_10_percent["text"].to_list()
# train_labels_10_percent = train_10_percent["target"].to_list()

#Making data split like this lead to data leakage, model_7 10 percent data outperforms
#model_6 data..
#DO NOT MAKE SPLITS WHICH LEAK DATA FROM VALIDATION/TEST SETS INTO TRAINNG SET

#MAKE BETTER DATA SET SPLIT WITHOUT LEAKAGE
train_10_percent_split = int(0.1 * len(train_sentences))
train_sentences_10_percent = train_sentences[:train_10_percent_split]
train_labels_10_percent = train_labels[:train_10_percent_split]
#be very careful when splitting data..(validation data set should not contains train data samples)

pd.Series(np.array(train_labels_10_percent)).value_counts()

#build model_7 same as model_6 we can use clone_model
model_7 = tf.keras.models.clone_model(model_6)
model_7.compile(loss="binary_crossentropy",
                optimizer=tf.keras.optimizers.Adam(),
                metrics=["accuracy"])

model_7.summary()

model_7_hist = model_7.fit(train_sentences_10_percent,
                           train_labels_10_percent,
                           epochs=5,
                           validation_data=(val_sentences, val_labels))

model_7_pred_probs = model_7.predict(val_sentences)
model_7_preds = tf.squeeze(tf.round(model_7_pred_probs))
model_7_preds[:10]

model_7_results = calculate_results(y_true=val_labels,
                                    y_pred=model_7_preds)



#comparing performance of each of our models
#combine model results into a dataframe 
all_model_results = pd.DataFrame({"baseline" : baseline_results,
                                  "1_simple_Dense" : model_1_results,
                                  "2_lstm" : model_2_results,
                                  "3_gru" : model_3_results,
                                  "4_bidirec" : model_4_results,
                                  "5_conv1d" : model_5_results,
                                  "6_tfhub" : model_6_results,
                                  "7_hub10percent" : model_7_results})

all_model_results = all_model_results.transpose()
#reduce the accuracy to the same scale as oyher metrics
all_model_results["accuracy"] = all_model_results["accuracy"]/100

#plot and compare all of model results
all_model_results.plot(kind="bar", figsize=(10,7)).legend(bbox_to_anchor=(1.0, 1.0))

#sort model results by f1 score
all_model_results.sort_values("f1", ascending=False)["f1"].plot(kind="bar")

#save and load trained model
#there are 2 main format to save model in tensorflow 
#1 - hdf5 #2 savedmodel format 
#save model 6 
model_6.save("model_6.h5")
#load model
import tensorflow_hub as hub
loaded_model = tf.keras.models.load_model("model_6.h5",
                                          custom_objects={"KerasLayer" : hub.KerasLayer})

#how does loaded model perform
loaded_model.evaluate(val_sentences, val_labels)

#saved model format
model_6.save("model_6_savedmodel_format")

#load model from savedmodel format
loaded_model_savedmodel_format = tf.keras.models.load_model("model_6_savedmodel_format")

loaded_model_savedmodel_format.evaluate(val_sentences, val_labels)

#find most wrong examples
#if our best model still isnt perfect what examples getting wrong??

#create dataframe with validation sentences and best performin model predictions


#unzip_data("08_model_6_USE_feature_extractor.zip")
model_6_pretrained = tf.keras.models.load_model("08_model_6_USE_feature_extractor")
model_6_pretrained.evaluate(val_sentences, val_labels)


#make predictions with model
model_6_pretrained_pred_probs = model_6_pretrained.predict(val_sentences)
model_6_pretrained_preds = tf.squeeze(tf.round(model_6_pred_probs))
model_6_pretrained_preds[:5]

val_df = pd.DataFrame({"text" : val_sentences,
                       "target" : val_labels,
                       "pred" : model_6_pretrained_preds,
                       "pred_prob" : tf.squeeze(model_6_pretrained_pred_probs)})

val_df

#we want to find most wrong predictions
most_wrong = val_df[val_df["target"] != val_df["pred"]].sort_values("pred_prob", ascending=False)

for row in most_wrong[:10].itertuples():
    _, text, target, pred, pred_prob = row
    print(f"Target:{target}, Pred:{pred}, Prob:{pred_prob}")
    print(f"Text:\n{text}\n")
    print("----\n")


for row in most_wrong[-10:].itertuples():
    _, text, target, pred, pred_prob = row
    print(f"Target:{target}, Pred:{pred}, Prob:{pred_prob}")
    print(f"Text:\n{text}\n")
    print("----\n")

#make predictions on the test dataset..
test_sentences = test_df["text"].to_list()
test_sentences[:5]

test_samples = random.sample(test_sentences, 10)
for test_sample in test_samples:
    pred_prob = tf.squeeze(model_6_pretrained.predict([test_sample]))
    pred = tf.round(pred_prob)
    print(f"Pred: {int(pred)}, Prob:{pred_prob}")
    print(f"Text: \n{test_sample}\n")
    print("---\n")


#the speed/score tradeoff
#make function to time of prediction
import time

def pred_timer(model, samples):
    start_time = time.perf_counter()
    model.predict(samples)
    end_time = time.perf_counter()
    total_time = end_time - start_time
    time_per_prediction = total_time / len(samples)
    return total_time, time_per_prediction


#calculate tf hub encoder
model_6_total_pred_time, model_6_time_per_pred = pred_timer(model=model_6_pretrained,
                                                            samples=val_sentences)


model_6_total_pred_time, model_6_time_per_pred

#calculate baseline times per pred
baseline_total_pred_time, baseline_time_per_pred = pred_timer(model=model_0, 
                                                              samples=val_sentences)

baseline_total_pred_time, baseline_time_per_pred


model_6_pretrained_results = calculate_results(y_true=val_labels,
                                               y_pred = model_6_pretrained_preds)
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 7))
plt.scatter(baseline_time_per_pred, baseline_results["f1"], label="baseline")
plt.scatter(model_6_time_per_pred, model_6_pretrained_results["f1"], label="model6pretraiend")
plt.legend()
plt.title("f1score v time per prediction")





















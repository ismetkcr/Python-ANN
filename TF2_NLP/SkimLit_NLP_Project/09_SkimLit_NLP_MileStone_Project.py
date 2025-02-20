# -*- coding: utf-8 -*-
"""
Created on Sun Jan 19 12:17:54 2025

@author: ismt
"""

#purpose of this is to build NLP model make reading paper abstracts easier

# #get data
from helper_functions import unzip_data, list_items_in_path, calculate_results
# unzip_data("pubmed-rct-master.zip")

#checkk what files in pubmed_20k dataset
import os
data_dir = "pubmed-rct-master/PubMed_20k_RCT_numbers_replaced_with_at_sign/"
list_items_in_path(data_dir)

#start out experiments using 20k dataset with numbers replaces by @ sign
filenames = [data_dir + filename for filename in os.listdir(data_dir)]
filenames

#visualize examples from the dataset
def get_lines(filename):
    with open(filename, "r") as f:
        return f.readlines()

#read in train lines
train_lines = get_lines(data_dir+"train.txt")
train_lines[:20]
len(train_lines)

# #how we want our data to look?
# [{'line_number' : 0,
#   'target' : 'Background',
#   'text' : "......",
#   'total_lines' :11}]


def preprocess_text_with_line_numbers(filename):
    input_lines = get_lines(filename) #get all lines from filename
    abstract_lines="" #create empty abstarct
    abstract_samples = []
    
    for line in input_lines:
        if line.startswith('###'): #check to see if tis an ID line
            abstract_id = line
            abstract_lines="" #reset the abstract string if the line is ID line
        elif line.isspace(): #check to ifline is new line
            abstract_line_split = abstract_lines.splitlines() #split abstract into seperate lines
            #itereate through each line in single abstract and count them
            for abstract_line_number, abstract_line in enumerate(abstract_line_split):
                line_data = {} #create empty dict for each line
                target_text_split = abstract_line.split("\t") #split target label from text
                line_data["target"] = target_text_split[0]
                line_data["text"] = target_text_split[1].lower()
                line_data["line_number"] = abstract_line_number
                line_data["total_lines"] = len(abstract_line_split) - 1
                abstract_samples.append(line_data)
                
        else: #if the above conditions arent fulfilled the line contains labelled sentence
            abstract_lines += line
            
    return abstract_samples
    
train_samples = preprocess_text_with_line_numbers(data_dir + "train.txt")
val_samples = preprocess_text_with_line_numbers(data_dir + "dev.txt")
test_samples = preprocess_text_with_line_numbers(data_dir + "test.txt")
len(train_samples), len(val_samples), len(test_samples)           
            
import pandas as pd
#now our data form of list of dict, turn it into dataframe to further visualize it
train_df =  pd.DataFrame(train_samples)
val_df = pd.DataFrame(val_samples)
test_df = pd.DataFrame(test_samples)
train_df.head(14)

#distribution of labels check balance
train_df["target"].value_counts()

#check length of different lines
train_df.total_lines.plot.hist()

#get a list of sentences
#convert abstarct text lines into list 
train_sentences = train_df.text.to_list()
val_sentences = val_df.text.to_list()
test_sentences = test_df.text.to_list()
len(train_sentences), len(val_sentences), len(test_sentences)
train_sentences[:10]

#turning labels into numbers
#we need require numeric labels for sure.:D make it 
#one hot encode
from sklearn.preprocessing import OneHotEncoder
import tensorflow as tf
one_hot_encoder = OneHotEncoder(sparse=False)
train_labels_one_hot = one_hot_encoder.fit_transform(train_df.target.to_numpy().reshape(-1, 1))
tf.constant(train_labels_one_hot)
val_labels_one_hot = one_hot_encoder.fit_transform(val_df.target.to_numpy().reshape(-1, 1))
test_labels_one_hot = one_hot_encoder.fit_transform(test_df.target.to_numpy().reshape(-1, 1))

#extract labels ("target cols") and encode them into integers
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
train_labels_encoded = label_encoder.fit_transform(train_df.target.to_numpy())
val_labels_encoded = label_encoder.fit_transform(val_df.target.to_numpy())
test_labels_encoded = label_encoder.fit_transform(test_df.target.to_numpy())

#check what train labels looklike
train_labels_encoded

#get class names and num of classes from label encoder instance
num_classes = len(label_encoder.classes_)
class_names = label_encoder.classes_
num_classes, class_names

#MODEL 0 Naive Bayes with TF-IDF encoder (baseline)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

model_0 = Pipeline([
    ("tf-idf", TfidfVectorizer()),
    ("clf", MultinomialNB())
    ])

#fit the train data
model_0.fit(X=train_sentences,
            y=train_labels_encoded)

#evaluate
model_0.score(X=val_sentences,
                 y=val_labels_encoded)

#make predictions with trained model
baseline_preds = model_0.predict(val_sentences)
baseline_preds, val_labels_encoded

from helper_functions import calculate_results
baseline_results = calculate_results(y_true=val_labels_encoded,
                                    y_pred=baseline_preds)

#prepare data for Conv1D with token ebmedding model
#vectorization and embedding
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

#how long each sentence on average??
sent_lens = [len(sentences.split()) for sentences in train_sentences]
avg_sent_len = np.mean(sent_lens)

import matplotlib.pyplot as plt
plt.hist(sent_lens, bins=8);

#how long of a sentence length covers %95 of examples?
output_seq_len = int(np.percentile(sent_lens, 95))
#max seq len in train set
max(sent_lens)

#create textvectorizer
from tensorflow.keras.layers import TextVectorization
#V : max vocab size in vocabulary
max_tokens = 68000
text_vectorizer = TextVectorization(max_tokens=max_tokens,
                                 output_sequence_length=output_seq_len)

#adapt text vectorizer to train sentences
text_vectorizer.adapt(train_sentences)

#test out text vectorizer on random sentences
import random
target_sentence = random.choice(train_sentences)
print(f"TExt:\n{target_sentence}")
print(f"\nLength of text: {len(target_sentence.split())}")
print(f"\nVectorized text: {text_vectorizer([target_sentence])}")

#how many words in our train vocab
rct_20k_text_vocab = text_vectorizer.get_vocabulary()
print(f"Number of words in vocab: {len(rct_20k_text_vocab)}")
print(f"\nMost Common words in vocab: {rct_20k_text_vocab[:5]}")
print(f"\nMost Less words in vocab: {rct_20k_text_vocab[-5:]}")

#check vectorizer parameters
text_vectorizer.get_config()

#Embedding..
from tensorflow.keras.layers import Embedding
token_embed = Embedding(
    input_dim=len(rct_20k_text_vocab),
    output_dim=128,
    mask_zero=True, #its good variable len input 
    name="token_embedding")
#different embeddign sized result in drastically different numbers to train

#example of embedding
print(f"Sentence before vectorization: \n {target_sentence}\n")
vectorized_sentence = text_vectorizer([target_sentence])
print(f"Sentence after vectorization (before embedding):\n {vectorized_sentence}\n")
embedded_sentence = token_embed(vectorized_sentence)
print(f"Sentence after embedding:\n {embedded_sentence}")
print(f"Shape of embedding : {embedded_sentence.shape}")

#create fast loading data set fit tf.data API (better performance)

train_dataset = tf.data.Dataset.from_tensor_slices((train_sentences, train_labels_one_hot))
valid_dataset = tf.data.Dataset.from_tensor_slices((val_sentences, val_labels_one_hot))
test_dataset = tf.data.Dataset.from_tensor_slices((test_sentences, test_labels_one_hot))

#prefetch and batch data #currently i am not using GPU..
train_dataset = train_dataset.batch(32).prefetch(tf.data.AUTOTUNE)
valid_dataset = valid_dataset.batch(32).prefetch(tf.data.AUTOTUNE)
test_dataset = test_dataset.batch(32).prefetch(tf.data.AUTOTUNE)


#Model - 1 with Conv1D layer
inputs = tf.keras.layers.Input(shape=(1,), dtype=tf.string)
text_vectors = text_vectorizer(inputs) #vectorize text to integers
token_embeddings = token_embed(text_vectors) #embedding
x = layers.Conv1D(64, kernel_size=5, activation="relu", padding="same")(token_embeddings)
x = layers.GlobalAveragePooling1D()(x)
outputs = layers.Dense(num_classes, activation="softmax")(x)
model_1 = tf.keras.models.Model(inputs, outputs)

model_1.compile(loss="categorical_crossentropy",
                optimizer=tf.keras.optimizers.Adam(),
                metrics=["accuracy"])

history_model_1 = model_1.fit(train_dataset,
                              steps_per_epoch=int(0.1*len(train_dataset)),
                              epochs=3,
                              validation_data=(valid_dataset),
                              validation_steps=int(0.1*len(valid_dataset))
                              )

#evaluate on whole valudation set
model_1.evaluate(valid_dataset)

#make predictions
model_1_pred_probs = model_1.predict(valid_dataset)
model_1_preds = model_1_pred_probs.argmax(axis=1)

model_1_results = calculate_results(y_true=val_labels_encoded,
                                    y_pred=model_1_preds)
 

#model 2 feature extraction with pretrained token embeddings
#download pre-trained USE
import tensorflow_hub as hub
tf_hub_embed_layer = hub.KerasLayer("https://tfhub.dev/google/universal-sentence-encoder/4",
                                    trainable=False,
                                    name="USE")

#test pretraiend embedding
random_train_sentence = random.choice(train_sentences)
print(f"Random sentence: \n {random_train_sentence}")
use_embedded_sentence = tf_hub_embed_layer([random_train_sentence])
print(f"Sentence after embedding: \n {use_embedded_sentence} " )
len(use_embedded_sentence)

#define feature extraction model 
inputs = layers.Input(shape=[], dtype=tf.string)
pretrained_embedding = tf_hub_embed_layer(inputs) #tokenizes automatically and create embedding
x = layers.Dense(128, activation="relu")(pretrained_embedding)
#we can add more layers here if we want ?
outputs = layers.Dense(num_classes, activation="softmax")(x)
model_2 = tf.keras.models.Model(inputs, outputs,
                                name="model_2_USE")

model_2.compile(loss="categorical_crossentropy",
                optimizer=tf.keras.optimizers.Adam(),
                metrics=["accuracy"])

model_2.summary()

history_model_2 = model_2.fit(train_dataset,
                              epochs=3,
                              steps_per_epoch=int(0.1*len(train_dataset)),
                              validation_data=valid_dataset,
                              validation_steps=int(0.1*len(valid_dataset)))

#evaluate whloe valid set
model_2.evaluate(valid_dataset)

#make predicts
model_2_pred_probs = model_2.predict(valid_dataset)
model_2_preds = model_2_pred_probs.argmax(axis=1)

model_2_results = calculate_results(y_true=val_labels_encoded,
                                    y_pred=model_2_preds)

#create character level tokenizer with TextVectorization layer
#previously token-level embedding now, char-level embedding
### create char-level tokenizer
train_sentences[:0]
" ".join(list(train_sentences[0]))


#make function to split sentences to char
def split_chars(text):
    return " ".join(list(text))

" ".join(list(train_sentences[0]))

split_chars(random_train_sentence)

train_chars = [split_chars(sentence) for sentence in train_sentences]
val_chars = [split_chars(sentence) for sentence in val_sentences]
test_chars = [split_chars(sentence) for sentence in test_sentences]

#find avg char len
char_lens = [len(sentence) for sentence in train_sentences]
mean_char_len = np.mean(char_lens)

#check distribution of sequences
import matplotlib.pyplot as plt
plt.hist(char_lens, bins=7)

#find what char lens covers &95 of sentences
output_seq_char_len = int(np.percentile(char_lens, 95))
#tensorflow text vectorization
import string
alphabet = string.ascii_lowercase + string.digits + string.punctuation
alphabet

#create char level tokenizer instance
num_char_tokens = len(alphabet) + 2 #add 2 for space and OOV token (out of token)
char_vectorizer = TextVectorization(max_tokens=num_char_tokens,
                                    output_sequence_length=output_seq_char_len,
                                    name="char_vectorizer")

#adapt character vectorizer to train charactare
char_vectorizer.adapt(train_chars)
#check 
char_vocab = char_vectorizer.get_vocabulary()
char_vectorizer.get_config()
print(f"Numfer of different characters in char vocab : {len(char_vocab)}")
print(f"5 most common : {char_vocab[:5]}")
print(f"5 most least : {char_vocab[-5:]}")

#test
random_train_chars = random.choice(train_chars)
print(f"Charified text:\n {random_train_chars}")
print(f"length of random_train_chars:\n {len(random_train_chars.split())}")
vectorized_chars = char_vectorizer([random_train_chars])
print(f"Vectorized_chars: \n {vectorized_chars}")
print(f"\nLength of vectorized chars: {len(vectorized_chars[0])}")

#create character level embeddig..
char_embed = layers.Embedding(input_dim=len(char_vocab),
                              output_dim=25,
                              mask_zero=True,
                              name="char_embed")

#test
print(f"Charified text:\n {random_train_chars}\n")
char_embed_exp = char_embed(char_vectorizer([random_train_chars]))
print(f"Embedded chars after vect and embed\n {char_embed_exp}")
print(f"Char embed shape is : \n {char_embed_exp.shape}")

#model 3 Conv1D model with char embed
inputs = layers.Input(shape=(1,), dtype=tf.string)
char_vectors = char_vectorizer(inputs)
char_embeddings = char_embed(char_vectors)
x = layers.Conv1D(64, kernel_size=5, padding="same", activation="relu")(char_embeddings)
x = layers.GlobalMaxPooling1D()(x)
outputs = layers.Dense(num_classes, activation="softmax")(x)
model_3 = tf.keras.models.Model(inputs, outputs, name="conv1d_char_embed")

model_3.compile(loss="categorical_crossentropy",
                optimizer=tf.keras.optimizers.Adam(),
                metrics=["accuracy"])

model_3.summary()

#create char level datasets
train_char_dataset = tf.data.Dataset.from_tensor_slices((train_chars, train_labels_one_hot)).batch(32).prefetch(tf.data.AUTOTUNE)
val_char_dataset = tf.data.Dataset.from_tensor_slices((val_chars, val_labels_one_hot)).batch(32).prefetch(tf.data.AUTOTUNE)
test_char_dataset = tf.data.Dataset.from_tensor_slices((test_chars, test_labels_one_hot)).batch(32).prefetch(tf.data.AUTOTUNE)

model_3_history = model_3.fit(train_char_dataset,
                              steps_per_epoch=int(0.1*len(train_char_dataset)),
                              epochs=3,
                              validation_data=(val_char_dataset),
                              validation_steps=int(0.1*len(val_char_dataset)))

model_3_pred_probs = model_3.predict(val_char_dataset)
model_3_preds = model_3_pred_probs.argmax(axis=1)

model_3_results = calculate_results(y_true=val_labels_encoded,
                                    y_pred=model_3_preds)

#model 4 Combined pretrained token embed + char embed
#1. create token-level embedding
#2. create character-lebel embedding 
#3. combine both of this with concetenate
#4. build a series of output layers on top of 3
#5. Construct model with takes token and char levbel sequences as input 

#STEP 1 .. setup token inputs/model
token_inputs = layers.Input(shape=[], dtype=tf.string, name="token_input")
token_embeddings = tf_hub_embed_layer(token_inputs)
token_output = layers.Dense(128, activation="relu")(token_embeddings)
token_model = tf.keras.models.Model(inputs=token_inputs,
                                    outputs=token_output)

#STEP 2.. create character level token
char_inputs = layers.Input(shape=(1,), dtype=tf.string, name="char_input")
char_vectors = char_vectorizer(char_inputs)
char_embeddings = char_embed(char_vectors)
char_bi_lstm = layers.Bidirectional(layers.LSTM(24))(char_embeddings)
char_model = tf.keras.models.Model(inputs=char_inputs,
                                   outputs=char_bi_lstm)

#STEP 3.. Combine 1 and 2 concatenate token and char input 
token_char_concat = layers.Concatenate(name="token_char_hybrid")([token_model.output,
                                                                 char_model.output])

#STEP 4: Create output layers adding dropout 
combined_dropout = layers.Dropout(0.5)(token_char_concat)
combined_dense = layers.Dense(128, activation="relu")(combined_dropout)
final_dropout = layers.Dropout(0.5)(combined_dense)
output_layer = layers.Dense(num_classes, activation="softmax")(final_dropout)

#STEP 5 construct model wih char and token inputs
model_4 = tf.keras.Model(inputs=[token_model.input,
                                 char_model.input], #the order of input data is important, make same data training phase
                         outputs=output_layer,
                         name="model_4_token_and_char_embeddings")

model_4.summary()

from keras.utils import plot_model
plot_model(model_4)

model_4.compile(loss="categorical_crossentropy",
                optimizer=tf.keras.optimizers.Adam(),
                metrics=["accuracy"])

##Combining token and char data into tf.data.Dataset
train_token_char_data = tf.data.Dataset.from_tensor_slices((train_sentences, train_chars)) #this is missing label #order is important it should be same as model
train_token_char_labels = tf.data.Dataset.from_tensor_slices((train_labels_one_hot)) #make label
train_token_char_dataset = tf.data.Dataset.zip((train_token_char_data, train_token_char_labels))

train_token_char_dataset = train_token_char_dataset.batch(32).prefetch(tf.data.AUTOTUNE)
#repeat above steps for validation_Data
##Combining token and char data into tf.data.Dataset
val_token_char_data = tf.data.Dataset.from_tensor_slices((val_sentences, val_chars)) #this is missing label
val_token_char_labels = tf.data.Dataset.from_tensor_slices((val_labels_one_hot)) #make label
val_token_char_dataset = tf.data.Dataset.zip((val_token_char_data, val_token_char_labels))

val_token_char_dataset = val_token_char_dataset.batch(32).prefetch(tf.data.AUTOTUNE)


#check datasets
train_token_char_dataset, val_token_char_dataset

#fitting model on token and char level sequences

history_model_4 = model_4.fit(train_token_char_dataset,
                              steps_per_epoch=int(0.1*len(train_token_char_dataset)),
                              validation_data=(val_token_char_dataset),
                              epochs=3,
                              validation_steps=int(0.1*len(val_token_char_dataset)))

model_4.evaluate(val_token_char_dataset)
#make predictions
model_4_pred_probs = model_4.predict(val_token_char_dataset)
model_4_probs = model_4_pred_probs.argmax(axis=1)

model_4_results = calculate_results(y_true=val_labels_encoded,
                                    y_pred=model_4_probs)

##Model_5 transfer learning with pre trained token embeddings + character embeddings + POSITIONAL EMBEDDINGS now
train_df.head() #we need to extract line number and total lines
#any engineered features used to train model need to be avaible @ test time
#create positional embeddings
#how many different line numbers ?
train_df.line_number.value_counts()

#check distribution of line numbner
train_df.line_number.plot.hist()
#using tf for create one-hot encoded tensors of line_number 
train_line_numbers_one_hot = tf.one_hot(train_df.line_number.to_numpy(), depth=15)
val_line_numbers_one_hot = tf.one_hot(val_df.line_number.to_numpy(), depth=15)
test_line_numbers_one_hot = tf.one_hot(test_df.line_number.to_numpy(), depth=15)
train_line_numbers_one_hot[:10], train_line_numbers_one_hot.shape

#one_hot_encoded total_lines
train_df.total_lines.value_counts()
train_df.total_lines.plot.hist()
#check coverage of total lines value for cover %95
np.percentile(train_df.total_lines, 95)
train_total_lines_one_hot = tf.one_hot(train_df.total_lines.to_numpy(), depth=20)
val_total_lines_one_hot = tf.one_hot(val_df.total_lines.to_numpy(), depth=20)
test_total_lines_one_hot = tf.one_hot(test_df.total_lines.to_numpy(), depth=20)
train_total_lines_one_hot[:10], train_total_lines_one_hot


#MODEL-5 Build of tribrid embedding model
#1. Create token level model
#2. Create char level model
#3. Create line_number feature
#4. Create total_lines feature
#5. combine outputs of 1 and 2 using layers.Concatenate
#6. combine the outputs of 3, 4, 5 using concatenate
#7. create output layer to expect tribried embedding and output label probabiliies
#8 combine 1, 2, 3, 4 and outputs of into tf.keras.Model

#1 - token inpts
token_inputs = tf.keras.layers.Input(shape=[], dtype=tf.string, name="token_inputs")
token_embeddings = tf_hub_embed_layer(token_inputs)
token_outputs = layers.Dense(128, activation="relu")(token_embeddings)
token_model = tf.keras.models.Model(inputs=token_inputs, outputs=token_outputs)

#2 - char inputs
char_inputs = layers.Input(shape=(1,), dtype=tf.string)
char_vectors = char_vectorizer(char_inputs)
char_embeddings = char_embed(char_vectors)
char_bi_lstm = layers.Bidirectional(layers.LSTM(24))(char_embeddings)
char_model = tf.keras.models.Model(inputs=char_inputs, outputs=char_bi_lstm)

#3. 
line_number_inputs = layers.Input(shape=(15,), dtype=tf.float32)
x = layers.Dense(32, activation="relu")(line_number_inputs)
line_number_model = tf.keras.models.Model(inputs=line_number_inputs, outputs=x)

#4.

total_lines_inputs = layers.Input(shape=(20,), dtype=tf.float32)
y = layers.Dense(32, activation="relu")(total_lines_inputs)
total_line_model = tf.keras.Model(inputs=total_lines_inputs, outputs=y)

#5. combine token and char embeddings
combined_embeddings = layers.Concatenate()([token_model.output,
                                            char_model.output])

z = layers.Dense(256, activation="relu")(combined_embeddings)
z = layers.Dropout(0.5)(z)


#6 combine position embeddings with token char embeddings
tribrid_embeddings = layers.Concatenate()([line_number_model.output,
                                           total_line_model.output,
                                           z])

#7 create output layer
output_layer = layers.Dense(5, activation="softmax")(tribrid_embeddings)

#8 put together all
model_5 = tf.keras.models.Model(inputs=[
    line_number_model.input,
    total_line_model.input,
    token_model.input,
    char_model.input
    ],
    outputs=output_layer)

model_5.summary()

#compile model
model_5.compile(loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.2),
                optimizer=tf.keras.optimizers.Adam(),
                metrics=["accuracy"])

#label smoothing helps to prevent over-fitting 
#for examole if our model gets too confident on single class, it may stuck on that class and not consider others..
#rellay confident : [0, 0, 0, 1, 0]
#label smoothing does is it assigns some of the value from the highest prob to other classes
#[0.1, 0.1, 0.11, 0.97, 0.01]

#create tribrid embedding data with tf.data
train_char_token_pos_data = tf.data.Dataset.from_tensor_slices((train_line_numbers_one_hot,
                                                                train_total_lines_one_hot,
                                                                train_sentences,
                                                                train_chars))

train_char_token_pos_labels = tf.data.Dataset.from_tensor_slices(train_labels_one_hot)
train_char_token_pos_dataset = tf.data.Dataset.zip((train_char_token_pos_data, train_char_token_pos_labels))
train_char_token_pos_dataset = train_char_token_pos_dataset.batch(32).prefetch(tf.data.AUTOTUNE)

#create valid dataset
#create tribrid embedding data with tf.data
val_char_token_pos_data = tf.data.Dataset.from_tensor_slices((val_line_numbers_one_hot,
                                                                val_total_lines_one_hot,
                                                                val_sentences,
                                                                val_chars))

val_char_token_pos_labels = tf.data.Dataset.from_tensor_slices(val_labels_one_hot)
val_char_token_pos_dataset = tf.data.Dataset.zip((val_char_token_pos_data, val_char_token_pos_labels))
val_char_token_pos_dataset = val_char_token_pos_dataset.batch(32).prefetch(tf.data.AUTOTUNE)


#check input shapes
train_char_token_pos_dataset, val_char_token_pos_dataset

#fit the model
history_5 = model_5.fit(train_char_token_pos_dataset,
                        steps_per_epoch=0.1*(len(train_char_token_pos_dataset)),
                        epochs=3,
                        validation_data=val_char_token_pos_dataset,
                        validation_steps=0.1*len(val_char_token_pos_dataset))

model_5_pred_probs = model_5.predict(val_char_token_pos_dataset)
model_5_preds = model_5_pred_probs.argmax(axis=1)

model_5_results = calculate_results(y_true=val_labels_encoded, 
                                    y_pred=model_5_preds)

#compare model resuls 
all_model_results = pd.DataFrame({"baseline" : baseline_results,
                                  "model_1" : model_1_results,
                                  "model_2" : model_2_results,
                                  "model_3" : model_3_results,
                                  "model_4" : model_4_results,
                                  "model_5" : model_5_results})

all_model_results = all_model_results.transpose()
all_model_results

#reduce accuracy to same scale with othe rmetrics
all_model_results.accuracy = all_model_results.accuracy/100
all_model_results.plot(kind="bar").legend(bbox_to_anchor=(1.0, 1.0))

#sort results by f1 score
all_model_results.sort_values("f1", ascending=True).f1.plot(kind="bar")

##save and load model..
model_5.save("skimlit_tribrid_model")

#load model
import tensorflow as tf
loaded_model = tf.keras.models.load_model("skimlit_tribrid_model")

#make predictions with loaded model on valid set
loaded_model_pred_probs = loaded_model.predict(val_char_token_pos_dataset)
loaded_model_preds = loaded_model_pred_probs.argmax(axis=1)

#results of loaded model
loaded_model_results = calculate_results(y_true=val_labels_encoded, 
                                         y_pred=loaded_model_preds)


#model_5_results == loaded_model_results #(returns true its okey..)


#1 turn the test data samples into tf.data Dataset (test_char_token_pos_dataset) and evaluate 
#2. Find the most wrong predictions from
#3. Make example predictions on RCT abstract
test_pos_char_token_data = tf.data.Dataset.from_tensor_slices((
    test_line_numbers_one_hot,
    test_total_lines_one_hot,
    test_sentences,
    test_chars
    ))

test_pos_char_token_labels = tf.data.Dataset.from_tensor_slices(test_labels_one_hot)
test_pos_char_token_dataset = tf.data.Dataset.zip((test_pos_char_token_data,
                                                   test_pos_char_token_labels))

test_pos_char_token_dataset = test_pos_char_token_dataset.batch(32).prefetch(tf.data.AUTOTUNE)

test_pred_probs = loaded_model.predict(test_pos_char_token_dataset)
test_preds = test_pred_probs.argmax(axis=1)

loaded_model_results = calculate_results(y_true=test_labels_encoded,
                                         y_pred=test_preds)

loaded_model_results


#find most wrong predictions
test_pred_classes = [label_encoder.classes_[pred] for pred in test_preds]

#fill data frame with new values
test_df["prediction"] = test_pred_classes
test_df["pred_prob"] = tf.reduce_max(test_pred_probs, axis=1).numpy()
test_df["correct"] = test_df.prediction == test_df.target
test_df.head(20)

#find top 100 most wrong samples
top_100_wrong = test_df[test_df.correct==False].sort_values("pred_prob", ascending=False)[:100]
top_100_wrong

for row in top_100_wrong[0:10].itertuples():
    _, target, text, line_number, total_lines, prediction, pred_prob, _ = row
    print(f"Target: {target}, Pred: {prediction}, Prob: {pred_prob}, Line number: {line_number}, Total lines: {total_lines}\n")
    print(f"Text:\n{text}\n")
    print("-----\n")
 

"""
If you use this, please cite:

Julia Ive, Natalia Viani, David Chandran, Andre Bittar and Sumithra Velupillai. KCL-Health-NLP@CLEF eHealth 2018 Task 1: ICD-10 Coding of French and Italian Death Certificates with Character-Level Convolutional Neural Networks. CLEF 2018 Evaluation Labs and Workshop: Online Working Notes, CEUR-WS, September, 2018.
"""


import pandas as pd
from keras import backend as K
from keras.layers import *
from keras.models import Sequential, Model
from keras.utils import Sequence
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import sys
import logging
import os
from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping
from sklearn.metrics import precision_recall_fscore_support
from sklearn.preprocessing import LabelEncoder
import random
from sklearn.cross_validation import train_test_split
from nltk import word_tokenize
import pickle
import time
import ast

random.seed(9)

#in files paths: takes the csv files for raw and computed versions, test, dev and train as provided by the organisers

#Reference: Neveol A, Robert A, Grippo F, Morgand C, Orsi C, Pelikan L, Ramadier L, Rey G, Zweigenbaum P. CLEF eHealth 2018 Multilingual Information Extraction task Overview: ICD10 Coding of Death Certificates in French, Hungarian and Italian. CLEF 2018 Evaluation Labs and Workshop: Online Working Notes, CEUR-WS, September, 2018.

in_test_raw = ''
in_dev_raw = ''
in_train_raw = ''

in_test_comp = ''
in_dev_comp = ''
in_train_comp = ''


# model type: if we use string-matched codes, if aligned or raw version, if IT vocabulary is used to extend French vocabulary (necessary if FR weights are further used for IT models)

codes = True
ali = True
it = False

#uncomment if it==True for creation of joint FR and IT vocabularies (necessary if FR weights are further used for IT models)

it_in_test_raw = ''
it_in_dev_raw = ''
it_in_train_raw = ''

it_in_test_comp = ''
it_in_dev_comp = ''
it_in_train_comp = ''

# put True to reuse the previously dumped to current folder the char level vocabs tokenizer.pickle and tokenizer_codes.pickle
loadVoc=False

# put True to reuse the previously dumped to current folder the label dictionaries ind2label.pickle and label2ind.pickle
loadLe=False

# set path to the weights pre-trained on another language (eg., pre-trained on FR for IT)
pretrained_weights=''

null_str = 'NULL'
dummy_line_num=111

# parameters for the network
# according to 3rd quartile stat
max_words = 6
max_chars = 49

# chars for max codes 20 chars = 5 codes
max_codes = 20
max_lines = 6

rnn_units = 300
emb_dim = 300
conv_units = 256
kernel_small=3
kernel_large=7
pool_size = 2
strides = 1

conv_activ='relu'
initializer_func = 'glorot_uniform'
output_activation='softmax'

# parameters for training
batch_size = 50
max_epochs = 100
patience=2
optimizer = 'adadelta'
loss = 'categorical_crossentropy'

sep_csv_coma=';'
sep_csv=';'

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# output epochs
cp = ModelCheckpoint("./weights/cnn-char.{epoch:02d}.hdf5",
                     monitor='val_loss',
                     #monitor='loss',
                     verbose=0,
                     # save_best_only=True,
                     save_weights_only=True,
                     mode='auto', period=1)

# mkdir if doesn't exist
if not os.path.exists('./weights'):
    os.makedirs('./weights/')

earlystop = EarlyStopping(monitor='val_loss', min_delta=0.001, patience=patience, \
                          verbose=1, mode='auto')


def test_model_doc_no_duplicates_ali(test_x, test_y, model, set_name, codes=False):

    if not os.path.exists('./test_res'):
        os.makedirs('./test_res/')
    logger.info('Predicting')

    if codes:
        result = model.predict([test_x[0][:, :, 7:7 + max_chars], test_x[1]])
    else:
        result = model.predict(test_x[0][:, :, 7:7 + max_chars])

    test_x = test_x[0]

    threshold_array = []
    threshold_res = []

    for i in range(result.shape[0]):
        # logger.info('Doc ' + str(i))
        for j in range(result.shape[1]):

            probs = result[i, j, :]
            line1 = test_x[i, j, 0:7]
            line2 = test_x[i, j, -2:]

            str1 = ";".join(str(n) for n in line1)
            str2 = ";".join(str(n) for n in line2)

            threshold_array.append(str1 +';'+ str2)
            threshold_res.append(np.argmax(probs))

    res_list = []
    res_list.append('DocID;YearCoded;Gender;Age;LocationOfDeath;LineID;RawText;IntType;IntValue;CauseRank;StandardText;ICD10')
    threshold_res = dedict(threshold_res)
    for x, y in zip(threshold_array, threshold_res):
        #if null_str not in x:
        res_list.append(x + ';;;' + y)
    logger.info('Saving test')
    np.savetxt('./test_res/' + set_name + '.csv', res_list, fmt='%s')


def text_class_doc():

    in_txt = Input(name='in_norm',
                    batch_shape=tuple([None, None, max_chars]), dtype='int32')

    emb_char = Embedding(len(char_ind) + 1,
                    emb_dim, name='emb_char')

    emb_seq = emb_char(in_txt)
    
    # a couple of conv layers for norm input

    conv1_norm =TimeDistributed(Conv1D(conv_units, kernel_size=kernel_small, kernel_initializer=initializer_func, activation=conv_activ, strides=strides, name='conv1_norm'))(emb_seq)
    pool1_norm = TimeDistributed(MaxPooling1D(pool_size=pool_size, name='pool1_norm'))(conv1_norm)

    flat = TimeDistributed(Flatten(name='flatten'))(pool1_norm)
    rnn = Bidirectional(GRU(rnn_units, return_sequences=True, name='decoder'))(flat)
    out_soft = TimeDistributed(Dense(label_count, kernel_initializer=initializer_func,
                        activation=output_activation, name='out_soft'))(rnn)


    model = Model(inputs=in_txt, outputs=out_soft)
    model.compile(optimizer=optimizer,
                  loss=loss,
                  metrics=['accuracy'])
    logger.info(model.summary())
    logger.info('Model Compiled.')

    return model


def text_class_doc_codes():

    in_txt = Input(name='in_norm',
                   batch_shape=tuple([None, None, max_chars]), dtype='int32')

    in_codes = Input(name='in_codes',
                     batch_shape=tuple([None, None, max_codes]), dtype='int32')

    emb_char = Embedding(len(char_ind) + 1,
                         emb_dim, name='emb_char')

    emb_seq = emb_char(in_txt)

    emb_code = Embedding(len(char_ind_codes) + 1,
                         emb_dim, name='emb_code')

    emb_code_out = emb_code(in_codes)

    # a couple of conv layers for norm input

    conv1_norm = TimeDistributed(
        Conv1D(conv_units, kernel_size=kernel_small, kernel_initializer=initializer_func, activation=conv_activ,
               strides=strides, name='conv1_norm'))(emb_seq)
    pool1_norm = TimeDistributed(MaxPooling1D(pool_size=pool_size, name='pool1_norm'))(conv1_norm)

    # a couple of conv layers for codes input

    conv1_codes = TimeDistributed(
        Conv1D(conv_units, kernel_size=kernel_small, kernel_initializer=initializer_func, activation=conv_activ,
               strides=strides, name='conv1_codes'))(emb_code_out)
    pool1_codes= TimeDistributed(MaxPooling1D(pool_size=pool_size, name='pool1_codes'))(conv1_codes)


    flat1 = TimeDistributed(Flatten(name='flatten1'))
    norm_flat = flat1(pool1_norm)

    flat2 = TimeDistributed(Flatten(name='flatten2'))
    codes_flat = flat2(pool1_codes)

    concat = concatenate([norm_flat, codes_flat], name='concat')

    rnn = Bidirectional(GRU(rnn_units, return_sequences=True, name='decoder'))(concat)
    out_soft = TimeDistributed(Dense(label_count, kernel_initializer=initializer_func,
                                     activation=output_activation, name='out_soft'))(rnn)

    model = Model(inputs=[in_txt, in_codes], outputs=out_soft)
    model.compile(optimizer=optimizer,
                  loss=loss,
                  metrics=['accuracy'])
    logger.info(model.summary())
    logger.info('Model Compiled.')

    return model


def prepare_data(loadVoc=False, loadLe=False, ali=False, it=False):

    x_train_prep, x_dev_prep, x_test_prep, y_train, y_dev, y_test, it_x_train, it_x_dev = get_data_per_doc(ali=ali, loadLe=loadLe, it=it)

    # prepare all data to create char level vocab
    if it:
        all_data = np.append(x_train_prep[0].flatten(), it_x_train[0].flatten())
    else:
        all_data = x_train_prep[0].flatten()

    all_data = [[x for x in y.lower()] for y in all_data]

    # index raw text
    global char_ind
    if loadVoc:
        tokenizer = pickle.load(open("tokenizer.pickle", "rb" ))
    else:
        tokenizer = Tokenizer(char_level=True, filters='')
        tokenizer.fit_on_texts(all_data)
        pickle.dump(tokenizer, open("tokenizer.pickle", "wb"))

    char_ind = tokenizer.word_index

    # index data
    x_train = vectorize_data(tokenizer, x_train_prep[0], max_chars, threedim=True)
    x_dev = vectorize_data(tokenizer, x_dev_prep[0], max_chars, threedim=True)

    if ali:

        x_test_txt = vectorize_data(tokenizer, x_test_prep[0][:, :, 6], max_chars, threedim=True)
        x_test_prep2 = np.append(x_test_prep[0][:, :, :7], x_test_txt, axis=-1)
        x_test = np.append(x_test_prep2, x_test_prep[0][:, :, -2:], axis=-1)

    else:

        x_test_txt = vectorize_data(tokenizer, x_test_prep[0][:, :, 3], max_chars, threedim=True)
        x_test = np.append(x_test_prep[0][:,:,:-1], x_test_txt, axis=-1)


    if it:
        all_data = np.append(x_train_prep[1].flatten(), it_x_train[1].flatten())
    else:
        all_data = x_train_prep[1].flatten()

    all_data = [[x for x in y.lower()] for y in all_data]

    # index codes text

    global char_ind_codes
    if loadVoc:
        tokenizer_codes = pickle.load(open("tokenizer_codes.pickle", "rb" ))
    else:
        tokenizer_codes = Tokenizer(char_level=True, filters='')
        tokenizer_codes.fit_on_texts(all_data)
        pickle.dump(tokenizer_codes, open("tokenizer_codes.pickle", "wb"))

    char_ind_codes = tokenizer_codes.word_index

    # index data
    x_train_codes = vectorize_data(tokenizer_codes, x_train_prep[1], max_codes, threedim=True)
    x_dev_codes = vectorize_data(tokenizer_codes, x_dev_prep[1], max_codes, threedim=True)
    x_test_codes = vectorize_data(tokenizer_codes, x_test_prep[1], max_codes, threedim=True)


    y_train = onehot(y_train.flatten(), label_count)
    y_dev = onehot(y_dev.flatten(), label_count)

    y_train = y_train.reshape((-1, max_lines, label_count))
    y_dev = y_dev.reshape((-1, max_lines, label_count))

    x_train = np.concatenate((x_train, x_train_codes), axis=2)
    x_dev = np.concatenate((x_dev, x_dev_codes), axis=2)
    #x_test = np.concatenate((x_test, x_test_prep[1]), axis=2)

    x_test = (x_test, x_test_codes)

    return x_train, x_dev, x_test, y_train, y_dev, y_test

def get_data_per_doc(loadLe=False, ali=False, it=False):

    global ind2label, label2ind
    if loadLe:
        ind2label = pickle.load(open("ind2label.pickle", "rb"))
        label2ind = pickle.load(open("label2ind.pickle", "rb"))

    else:

        ind2label = {}
        label2ind = {}

        label2ind['NULL'] = 0
        ind2label[0] = 'NULL'

    x_train, x_train_codes, y_train = tok_pad_doc_no_duplicates(in_train_raw, in_train_comp, ali=ali, loadLe=loadLe)
    x_dev, x_dev_codes, y_dev = tok_pad_doc_no_duplicates(in_dev_raw, in_dev_comp, ali=ali, loadLe=loadLe)

    it_x_train, it_y_train = [], []
    it_x_train_codes, it_x_dev_codes = [], []
    it_x_dev, it_y_dev = [], []

    if it:
        it_x_train, it_x_train_codes, it_y_train = tok_pad_doc_no_duplicates(it_in_train_raw, it_in_train_comp, ali=ali,loadLe=loadLe)
        it_x_dev, it_x_dev_codes, it_y_dev = tok_pad_doc_no_duplicates(it_in_dev_raw, it_in_dev_comp, ali=ali, loadLe=loadLe)


    if ali:

        x_test, x_test_codes, y_test = tok_pad_doc_test_no_duplicates_ali(in_test_comp)

    else:

        x_test, x_test_codes, y_test = tok_pad_doc_test_no_duplicates(in_test_raw, in_test_comp)

    all_labels = np.append(y_train, y_dev)
    all_labels = np.append(all_labels, it_y_train)
    all_labels = np.append(all_labels, it_y_dev)

    if not loadLe:
        extenddicts(all_labels, ind2label, label2ind)
        pickle.dump(ind2label, open("ind2label.pickle", "wb"))
        pickle.dump(label2ind, open("label2ind.pickle", "wb"))


    global label_count
    label_count = len(label2ind)

    y_train = applydict(y_train.flatten())
    y_dev = applydict(y_dev.flatten())

    y_train = y_train.reshape((-1, max_lines))
    y_dev = y_dev.reshape((-1, max_lines))

    x_train = np.array(x_train)
    y_train = np.array(y_train)

    return (x_train, x_train_codes), (x_dev, x_dev_codes), (x_test, x_test_codes), y_train, y_dev, y_test, (it_x_train, it_x_train_codes), (it_x_dev, it_x_dev_codes)


def extenddicts(crp, ind2label, label2ind):

    for i in crp:

        if i not in label2ind:
            label2ind[i] = len(label2ind)
            ind2label[len(ind2label)] = i


def dedict(crp):
    decoded = []

    for i in crp:

        if i in ind2label:
            decoded.append(ind2label[i])
        else:
            print i
            decoded.append('UNK')

    return np.array(decoded)


def applydict(crp):
    decoded = []

    for i in crp:

        if i in label2ind:
            decoded.append(label2ind[i])
        else:
            decoded.append(len(label2ind))

    return np.array(decoded)


def vectorize_data(tokenizer, x, numb_chars, threedim=False):

    max2d  = x.shape[1]
    # flatten as keras tokenizer eats only 2d array
    x = np.array(x)
    x = x.flatten()
    # create char array
    x = [[x for x in y.lower()] for y in x]

    #index & pad
    x = tokenizer.texts_to_sequences(x)
    x = np.array(pad_sequences(x, maxlen=numb_chars))
     
    if threedim:
        # reshape back to 3d
        x = x.reshape((-1, max2d, numb_chars))

    return x


def tok_pad_doc_no_duplicates(in_file_raw, in_file_comp, ali=False, gold=False, loadLe=False):
    
    if ali or gold:
        in_file_raw=in_file_comp

    df_raw = pd.read_csv(in_file_raw, sep=sep_csv_coma)
    df_comp = pd.read_csv(in_file_comp, sep=sep_csv)

    df_raw = df_raw.fillna(null_str)
    df_comp = df_comp.fillna(null_str)

    df_raw.sort_values(['DocID', 'LineID'], ascending=[True, True], inplace=True)
    df_comp.sort_values(['DocID', 'LineID'], ascending=[True, True], inplace=True)

    d_raw = df_raw.groupby('DocID')['LineID'].apply(list).to_dict()
    d_comp = df_comp.groupby('DocID')['LineID'].apply(list).to_dict()

    if gold:
        d_txt = df_raw.groupby('DocID')['StandardText'].apply(list).to_dict()
    else:
        d_txt = df_raw.groupby('DocID')['RawText'].apply(list).to_dict()

    d_codes = df_comp.groupby('DocID')['ICD10'].apply(list).to_dict()

    d_lookup = df_raw.groupby('DocID')['dictionary_lookup'].apply(list).to_dict()

    doc_count = len(d_raw.keys())

    # fill padded arrays with nulls

    ft = np.full((doc_count, max_lines), null_str, dtype=object)
    ft_codes = np.full((doc_count, max_lines, 1), null_str, dtype=object)
    labels = np.full((doc_count, max_lines), null_str, dtype=object)

    for i, did in enumerate(d_raw):

        doc_line_count = 0
        label_line_count = 0


        # for each line in raw

        for k, lid in enumerate(d_raw[did]):

            if doc_line_count < max_lines:

                ft[i, doc_line_count] = d_txt[did][k]
                doc_line_count += 1

                line_codes = ast.literal_eval(d_lookup[did][k])
                # get a merged line of codes for char-level encoding
                line_codes_line = "".join(str(x) for x in line_codes)
                ft_codes[i, k, 0] = line_codes_line

        for j, lcid in enumerate(d_comp[did]):

            if label_line_count < max_lines:

                labels[i, label_line_count] = d_codes[did][j]
                label_line_count += 1

    return ft, ft_codes, labels


def tok_pad_doc_test_no_duplicates(in_file_raw, in_file_comp):

    df_raw = pd.read_csv(in_file_raw, sep=sep_csv_coma)
    df_comp = pd.read_csv(in_file_comp, sep=sep_csv)

    df_raw = df_raw.fillna(null_str)
    df_comp = df_comp.fillna(null_str)

    df_raw.sort_values(['DocID', 'LineID'], ascending=[True, True], inplace=True)
    df_comp.sort_values(['DocID', 'LineID'], ascending=[True, True], inplace=True)

    d_raw = df_raw.groupby('DocID')['LineID'].apply(list).to_dict()
    d_comp = df_comp.groupby('DocID')['LineID'].apply(list).to_dict()

    d_txt = df_raw.groupby('DocID')['RawText'].apply(list).to_dict()
    d_year = df_raw.groupby('DocID')['YearCoded'].apply(list).to_dict()

    d_lookup = df_raw.groupby('DocID')['dictionary_lookup'].apply(list).to_dict()
    d_codes = df_comp.groupby('DocID')['ICD10'].apply(list).to_dict()

    # for test for take max lines not to miss anything
    lines_test = [len(x) for x in d_raw.values()]
    global max_lines_test
    max_lines_test = np.max(np.array(lines_test))+2

    doc_count = len(d_raw.keys())

    # we add also all line info to be able to print it out to csv later
    ft = np.full((doc_count, max_lines_test, 4), null_str, dtype=object)
    ft_codes = np.full((doc_count, max_lines_test, 1), null_str, dtype=object)

    labels = []

    for i, did in enumerate(d_raw):


        for k, lid in enumerate(d_raw[did]):

            doc_lid = []

            # we add all lines in raw

            ft[i, k, 0] = did
            ft[i, k, 1] = d_year[did][0]
            ft[i, k, 2] = lid
            ft[i, k, 3] = d_txt[did][k]

            line_codes = ast.literal_eval(d_lookup[did][k])
            line_codes_line = "".join(str(x) for x in line_codes)

            #for l in range(list_min):
            #   if line_codes[l] in label2ind:
            ft_codes[i, k, 0] = line_codes_line

            # for ref we add lines present in raw and computed
            if lid in d_comp[did]:

                for j,lcid in enumerate(d_comp[did]):

                    if lcid == lid:

                        doc_lid.append(did)
                        doc_lid.append(d_year[did][0])
                        doc_lid.append(lid)
                        doc_lid.append(d_codes[did][j])

                        labels.append(doc_lid)
                        doc_lid = []


        # for ref we add lines present only in computed

        for j, lcid in enumerate(d_comp[did]):

            if lcid not in d_raw[did]:

                doc_lid.append(did)
                doc_lid.append(d_year[did][0])
                doc_lid.append(lcid)
                doc_lid.append(d_codes[did][j])

                labels.append(doc_lid)

                doc_lid = []

        # at the end we add an empty line to be able to predict lables for lines present in comp but not in raw

        for j in range(k+1, max_lines_test):

            # fill with dummy values

            ft[i, j, 0] = did
            ft[i, j, 1] = d_year[did][0]
            #ft[i, j, 2] = lid+1
            ft[i, j, 2] = dummy_line_num
            ft[i, j, 3] = null_str

    return ft, ft_codes, labels

def tok_pad_doc_test_no_duplicates_ali(in_file):

    #TO DO: suboptimal procedure

    df_raw = pd.read_csv(in_file, sep=sep_csv)
    df_raw = df_raw.fillna(null_str)

    d_year = df_raw.groupby('DocID')['YearCoded'].apply(list).to_dict()
    d_gender = df_raw.groupby('DocID')['Gender'].apply(list).to_dict()
    d_age = df_raw.groupby('DocID')['Age'].apply(list).to_dict()
    d_loc = df_raw.groupby('DocID')['LocationOfDeath'].apply(list).to_dict()
    d_raw = df_raw.groupby('DocID')['LineID'].apply(list).to_dict()
    d_txt = df_raw.groupby('DocID')['RawText'].apply(list).to_dict()
    d_type = df_raw.groupby('DocID')['IntType'].apply(list).to_dict()
    d_value = df_raw.groupby('DocID')['IntValue'].apply(list).to_dict()

    d_lookup = df_raw.groupby('DocID')['dictionary_lookup'].apply(list).to_dict()

    # for test for take max lines not to miss anything
    lines_test = [len(x) for x in d_raw.values()]
    global max_lines_test
    max_lines_test = np.max(np.array(lines_test))

    doc_count = len(d_raw.keys())

    # we add also all line info to be able to print it out to csv later
    ft = np.full((doc_count, max_lines_test, 9), null_str, dtype=object)
    ft_codes = np.full((doc_count, max_lines_test, 1), null_str, dtype=object)


    for i, did in enumerate(d_raw):
        for k, lid in enumerate(d_raw[did]):

            ft[i, k, 0] = did
            ft[i, k, 1] = d_year[did][k]
            ft[i, k, 2] = d_gender[did][k]
            ft[i, k, 3] = d_age[did][k]
            ft[i, k, 4] = d_loc[did][k]
            ft[i, k, 5] = lid
            ft[i, k, 6] = d_txt[did][k]
            ft[i, k, 7] = d_type[did][k]
            ft[i, k, 8] = d_value[did][k]

            line_codes = ast.literal_eval(d_lookup[did][k])
            line_codes_line = "".join(str(x) for x in line_codes)
            #list_min = np.min([max_codes, len(line_codes)])

            #for l in range(list_min):
             #   if line_codes[l] in label2ind:
            ft_codes[i, k, 0] = line_codes_line

    return ft, ft_codes, []



def onehot(in_array, voc_size):
    res = []
    for i in range(len(in_array)):
        zero = np.zeros((voc_size,))
        zero[in_array[i]] = 1
        res.append(zero)

    return np.array(res)

# sequence data generator

class DataGenerator(Sequence):
    
    def __init__(self, data, labels, batch_size=batch_size,
                 n_classes=10, shuffle=True, threedim=False, doc=False, codes=False):
       
        self.batch_size = batch_size
        self.labels = labels
        self.data = data
        self.shuffle = shuffle
        self.threedim = threedim
        self.doc = doc
        self.codes = codes
        self.on_epoch_end()

    def __len__(self):
    
        return int(np.floor(len(self.indexes) / self.batch_size))

    def __getitem__(self, index):
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        # Generate data
        return self.__data_generation(indexes)

    def on_epoch_end(self):
        #updates indexes after each epoch
        self.indexes = np.arange(len(self.data))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        
        if self.threedim:
            X = np.empty((self.batch_size, self.data.shape[1], self.data.shape[2]))
            
            if self.doc:
                y = np.empty((self.batch_size, self.labels.shape[1], self.labels.shape[2]), dtype=int)
            else:
                y = np.empty((self.batch_size, self.labels.shape[1]), dtype=int)
        else:
            
            X = np.empty((self.batch_size, self.data.shape[1]))
            y = np.empty((self.batch_size, self.labels.shape[1]), dtype=int)

        for i,j in enumerate(list_IDs_temp):
            
            X[i,] = self.data[j,]
            y[i,] = self.labels[j,]

        if self.codes:
            return [X[:,:,0:max_chars], X[:,:,max_chars:]], y
        else:
            return X[:, :, 0:max_chars], y

if __name__ == '__main__':


    # get word indexes and indexed data
    logger.info('Indexing data')

    x_train, x_dev, x_test, y_train, y_dev, y_test = prepare_data(loadVoc=loadVoc, loadLe=loadLe, ali=ali, it=it)

    logger.info('Compiling model')

    if codes:
        model = text_class_doc_codes()
    else:
        model = text_class_doc()

    logger.info('Starting training')

    #load pre-trained weights if needed (eg. for italian)

    if pretrained_weights:
        model.load_weights(pretrained_weights, by_name=True)

    # data generators

    training_generator = DataGenerator(x_train, y_train, n_classes=label_count, threedim=True, doc=True, codes=codes)
    validation_generator = DataGenerator(x_dev, y_dev, n_classes=label_count, shuffle=False, threedim=True, doc=True, codes=codes)

    start = time.time()
    model.fit_generator(training_generator, validation_data=validation_generator, steps_per_epoch=int(np.floor(len(x_train) / batch_size)), validation_steps=int(np.floor(len(x_dev) / batch_size)), epochs=max_epochs, verbose=1, callbacks=[cp, earlystop], workers=6)

    end = time.time()
    print "Model took %0.2f seconds to train"%(end - start)
    best_epoch = str(earlystop.stopped_epoch)


    # load the best model for testing
    logger.info('Loading model '+ 'weights/cnn-char.'+best_epoch+'.hdf5')
    model.load_weights('weights/cnn-char.'+best_epoch+'.hdf5', by_name=True)
    logger.info('Model loaded')
    if ali:
        test_model_doc_no_duplicates_ali(x_test, y_test, model, 'test', codes=codes)
    else:
        test_model_doc_no_duplicates(x_test, y_test, model, 'test', codes=codes)



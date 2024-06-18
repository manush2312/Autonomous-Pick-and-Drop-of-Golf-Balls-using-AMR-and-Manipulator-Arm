import torch
import numpy as np
import re
import pickle

with open('./artifacts/vocab.pkl', 'rb') as f:
    vocab = pickle.load(f)

def padding_(sentences, seq_len):
    features = np.zeros((len(sentences), seq_len),dtype=int)
    for ii, review in enumerate(sentences):
        if len(review) != 0:
            features[ii, -len(review):] = np.array(review)[:seq_len]
    return features


def decode_to_text(encoded_text):
    decode = ''
    key_list = list(vocab.keys())
    val_list = list(vocab.values())
    for i in encoded_text:
        decode += (key_list[val_list.index(i)])
        decode += ' '
    return decode

onehot_dict = {}
def preprocess_string(s):
    s = re.sub(r"[^\w\s]", '', s)
    # Replace all runs of whitespaces with no space
    s = re.sub(r"\s+", '', s)
    # replace digits with no space
    s = re.sub(r"\d", '', s)
    
    return s

def tokenize(x_train, y_train, x_val, y_val):
    word_list = []
    
    stop_words = set(stopwords.words('english'))
    for sent in x_train:
        for word in sent.lower().split():
            word = preprocess_string(word)
            if word not in stop_words and word != '':
                word_list.append(word)
                
#     print(word_list)
    corpus = Counter(word_list)
    corpus_ = sorted(corpus, key=corpus.get, reverse=True)[:1000]
    
    #creating a dict
    onehot_dict = {w:i+1 for i,w in enumerate(corpus_)}
    
    #tockenize
    final_list_train, final_list_test = [], []
    for sent in x_train:
        print(sent)
        final_list_train.append([onehot_dict[preprocess_string(word)] for word in sent.lower().split() 
                                     if preprocess_string(word) in onehot_dict.keys()])
    for sent in x_val:
#         print(sent)
        final_list_test.append([onehot_dict[preprocess_string(word)] for word in sent.lower().split() 
                                if preprocess_string(word) in onehot_dict.keys()])
        
    encoded_train = [0 if label =='red' else(1 if label == 'green' else 2) for label in y_train]
    encoded_test = [0 if label =='red' else(1 if label == 'green' else 2) for label in y_val]
#     print(encoded_train)
    return np.array(final_list_train), np.array(encoded_train),np.array(final_list_test), np.array(encoded_test),onehot_dict


def predict_text(text):
    lstm_model = torch.jit.load('./model/golfball.pt')
    word_seq = np.array([vocab[preprocess_string(word)] for word in text.split() 
                     if preprocess_string(word) in vocab.keys()])
    word_seq = np.expand_dims(word_seq,axis=0)
    pad =  torch.from_numpy(padding_(word_seq,14))
    with torch.no_grad():
        lstm_model.eval()
        outputs = lstm_model(pad)

# Obtain predicted label
    predicted_label_index = torch.argmax(outputs, dim=1).item()

    # Map index to the label
    label_mapping = {0: 'red', 1: 'green', 2: 'blue'}
    predicted_label = label_mapping[predicted_label_index]

    return label_mapping, [0], predicted_label_index


def forward(input_text):
    parser = str(input_text)
    label_mapping, pro, out_class = predict_text(parser)
    print('The predicted class is ', label_mapping[out_class])
    return out_class


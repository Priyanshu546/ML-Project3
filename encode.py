# encoding the labels 
le = preprocessing.LabelEncoder() 
le.fit(data['label']) 
data['label'] = le.transform(data['label']) 
embedding_dim = 50
max_length = 54
trunc_type = 'post'
padding_type = 'post'
oov_tok = "<OOV>"
training_size = 3000
test_portion = .1

title = [] 
text = [] 
labels = [] 
for x in range(training_size): 
	title.append(data['title'][x]) 
	text.append(data['text'][x]) 
	labels.append(data['label'][x]) 
tokenizer1 = Tokenizer() 
tokenizer1.fit_on_texts(title) 
word_index1 = tokenizer1.word_index 
vocab_size1 = len(word_index1) 
sequences1 = tokenizer1.texts_to_sequences(title) 
padded1 = pad_sequences( 
	sequences1, padding=padding_type, truncating=trunc_type) 
split = int(test_portion * training_size) 
training_sequences1 = padded1[split:training_size] 
test_sequences1 = padded1[0:split] 
test_labels = labels[0:split] 
training_labels = labels[split:training_size] 

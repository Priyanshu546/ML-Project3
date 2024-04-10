model = tf.keras.Sequential([ 
	tf.keras.layers.Embedding(vocab_size1+1, embedding_dim, 
							input_length=max_length, weights=[ 
								embeddings_matrix], 
							trainable=False), 
	tf.keras.layers.Dropout(0.2), 
	tf.keras.layers.Conv1D(64, 5, activation='relu'), 
	tf.keras.layers.MaxPooling1D(pool_size=4), 
	tf.keras.layers.LSTM(64), 
	tf.keras.layers.Dense(1, activation='sigmoid') 
]) 
model.compile(loss='binary_crossentropy', 
			optimizer='adam', metrics=['accuracy']) 
model.summary() 
num_epochs = 50

training_padded = np.array(training_sequences1) 
training_labels = np.array(training_labels) 
testing_padded = np.array(test_sequences1) 
testing_labels = np.array(test_labels) 

history = model.fit(training_padded, training_labels, 
					epochs=num_epochs, 
					validation_data=(testing_padded, 
									testing_labels), 
					verbose=2) 
# sample text to check if fake or not 
X = "Karry to go to France in gesture of sympathy"

# detection 
sequences = tokenizer1.texts_to_sequences([X])[0] 
sequences = pad_sequences([sequences], maxlen=54, 
						padding=padding_type, 
						truncating=trunc_type) 
if(model.predict(sequences, verbose=0)[0][0] >= 0.5): 
	print("This news is True") 
else: 
	print("This news is false") 

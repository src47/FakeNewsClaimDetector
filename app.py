import streamlit as st 
import numpy as np 
import pandas as pd 
import matplotlib 
matplotlib.use('Agg')
import matplotlib.pyplot as plt 
import json
import requests
import base64
import tensorflow as tf 
from tensorflow import keras
import pickle
import os
from tensorflow.keras.preprocessing.sequence import pad_sequences
import random 
import SessionState
# Web scraping 
from bs4 import BeautifulSoup 
from urllib.request import urlopen 
import csv


os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "claimevaluatorcs329s-b77656fd4ca4.json" # change for your GCP key
PROJECT = "ClaimEvaluatorCS329S" # change for your GCP project
REGION = "us-west1" # change for your GCP region (where your model is hosted)

# API_TOKEN = "api_rkjoWdJwUXqOXbUXypHaZyYwzfiFRCDZcr"
# API_URL = "https://api-inference.huggingface.co/models/textattack/bert-base-uncased-imdb"
# headers = {"Authorization": f"Bearer {API_TOKEN}"}

# def query(payload):
#     data = json.dumps(payload)
#     response = requests.request("POST", API_URL, headers=headers, data=data)
#     return json.loads(response.content.decode("utf-8"))

# Get our Text
@st.cache
def get_text(raw_url):
	page = urlopen(raw_url)
	soup = BeautifulSoup(page)
	fetched = ' '.join(map(lambda p:p.text, soup.find_all('p')))
	return fetched

def load_tokenizer(filename):
	with open(filename, 'rb') as handle:
		tokenizer = pickle.load(handle)

	return tokenizer

def preprocess_text(tokenizer,text):
  """
  Args:
	tokenizer: keras tokenizer object
	text(list): list of strings of texts to make predictions on
  
  Return
	padded_sequence (np.array): shape [n_examples, maxlen] 

  """
  encoded_docs = tokenizer.texts_to_sequences(text)
  padded_sequence= pad_sequences(encoded_docs, maxlen=300)
  return padded_sequence


@st.cache
def load_testing_data(filename_data):
	with open(filename_data, 'rb') as handle:
		data = pickle.load(handle)

	return data



def main():

	label2class = {'False': 0, 'Mixture': 1, 'True': 2, 'Unproven': 3}

	outcomes = ["Article contains false information.", "Article contains some true and some false information.", "Article contains true information.", "Article contains unproven information."]    

	pubhealth = load_testing_data('test.data')
	
	st.title("Public Health Fake News Detector")

	options = st.sidebar.selectbox("Choose a page", ["User Article Input", "User Website Input", "Testing"])
	
	tokenizer = load_tokenizer('tokenizer.pkl')

	model = keras.models.load_model('lstm_token300_dim32_softmax.h5')


	if options == "User Article Input":

		st.subheader("Prediction on article")    
		text_box = st.text_area("Enter Text", "Type Here")

		session_state = SessionState.get(analyze_button=False)

		if text_box != 'Type Here':

		   
	
			model_selected = st.selectbox("Select a model", ['Baseline LSTM'])

			if st.button("Analyze"):

				session_state.analyze_button = True

			if session_state.analyze_button:

				X = preprocess_text(tokenizer,[text_box])

				# Prediction and print dataframe of probabilities 
				pred = model.predict(X)
				pred_df = pd.DataFrame(data=pred)
				pred_df.columns =['False', 'Mixture', 'True', 'Unproven'] 
				test_class = np.argmax(pred, axis = -1)
#                st.dataframe(pred_df.style.highlight_max(axis=1, color='green'))

				if test_class == 0:
					st.dataframe(pred_df.style.highlight_max(axis=1, color='red'))
					st.error(outcomes[np.int(test_class)])
				elif test_class == 1:
					st.dataframe(pred_df.style.highlight_max(axis=1, color='yellow'))
					st.warning(outcomes[np.int(test_class)])
				elif test_class == 2:
					st.dataframe(pred_df.style.highlight_max(axis=1, color='green'))
					st.success(outcomes[np.int(test_class)])
				elif test_class == 3:
					st.dataframe(pred_df.style.highlight_max(axis=1, color='grey'))
					st.info(outcomes[np.int(test_class)])
						
				user_eval = st.selectbox("Was this prediction correct?", ("Yes", "No", "Unsure"))
				
				if user_eval == "No":
					correct_label = label2class[st.selectbox("What is the correct label?", ('False', 'Mixture', 'True', 'Unproven'))]
					evidence = st.text_area("Evidence for Correct Label", "Type Here")
				elif user_eval == "Yes":
					correct_label = test_class[0]
					evidence = 'N/A'
				else:
					correct_label = 'N/A'
					evidence = 'N/A'

				saved_data = [text_box, user_eval, test_class[0], correct_label, evidence]

				if st.button("Submit") and session_state.analyze_button:

					st.write("User Evaluation Submitted!")

					with open("user.data.csv", "a") as f:
						wr = csv.writer(f, dialect='excel')
						wr.writerow(saved_data)

					session_state.analyze_button = False



	if options == "User Website Input":

		st.subheader("Prediction from URL")

		raw_url = st.text_input("Enter URL", "Type Here")
		session_state1 = SessionState.get(analyze_button=False)    
		#text_limit = st.slider("Length of Text to Provide", 50,90)
		if raw_url != 'Type Here':
			
			
			model_selected = st.selectbox("Select a model", ['Baseline LSTM'])

			if st.button("Analyze"):

				session_state1.analyze_button = True

			if session_state1.analyze_button:

				text = get_text(raw_url)

				st.write(' '.join(text.split(' ')[0:200]) + "...")
				X = preprocess_text(tokenizer,[text])

				pred = model.predict(X)
				pred_df = pd.DataFrame(data=pred)
				pred_df.columns =['False', 'Mixture', 'True', 'Unproven'] 


				#st.dataframe(pred_df.style.highlight_max(axis=1, color='green'))

				test_class = np.argmax(pred, axis = -1)

				if test_class == 0:
					st.dataframe(pred_df.style.highlight_max(axis=1, color='red'))
					st.error(outcomes[np.int(test_class)])
				elif test_class == 1:
					st.dataframe(pred_df.style.highlight_max(axis=1, color='yellow'))
					st.warning(outcomes[np.int(test_class)])
				elif test_class == 2:
					st.dataframe(pred_df.style.highlight_max(axis=1, color='green'))
					st.success(outcomes[np.int(test_class)])
				elif test_class == 3:
					st.dataframe(pred_df.style.highlight_max(axis=1, color='grey'))
					st.info(outcomes[np.int(test_class)])
						
				user_eval = st.selectbox("Was this prediction correct?", ("Yes", "No", "Unsure"))
				
				if user_eval == "No":
					correct_label = label2class[st.selectbox("What is the correct label?", ('False', 'Mixture', 'True', 'Unproven'))]
					evidence = st.text_area("Evidence for Correct Label", "Type Here")
				elif user_eval == "Yes":
					correct_label = test_class[0]
					evidence = 'N/A'
				else:
					correct_label = 'N/A'
					evidence = 'N/A'

				saved_data = [text, user_eval, test_class[0], correct_label, evidence]

				if st.button("Submit") and session_state1.analyze_button:

					st.write("User Evaluation Submitted!")

					with open("user.data.csv", "a") as f:
						wr = csv.writer(f, dialect='excel')
						wr.writerow(saved_data)

					session_state1.analyze_button = False



	if options == "Testing":

		st.subheader("Prediction on testing data")

		session_state = SessionState.get(checkboxed=False)
		model_selected = st.selectbox("Select a model", ['Baseline LSTM'])

		user_input = st.number_input("Choose a test set example", 0, len(pubhealth)-1)

		choice = pubhealth[user_input]
		st.write(choice)

		text = choice['main_text']


		if st.button("Analyze"):
			X = preprocess_text(tokenizer,[text])
			pred = model.predict(X)
			pred_df = pd.DataFrame(data=pred)
			pred_df.columns =['False', 'Mixture', 'True', 'Unproven'] 
 #           st.dataframe(pred_df.style.highlight_max(axis=1, color='green'))


			test_class = np.argmax(pred, axis = -1)

			if test_class == 0:
				st.dataframe(pred_df.style.highlight_max(axis=1, color='red'))
				st.error(outcomes[np.int(test_class)])
			elif test_class == 1:
				st.dataframe(pred_df.style.highlight_max(axis=1, color='yellow'))
				st.warning(outcomes[np.int(test_class)])
			elif test_class == 2:
				st.dataframe(pred_df.style.highlight_max(axis=1, color='green'))
				st.success(outcomes[np.int(test_class)])
			elif test_class == 3:
				st.dataframe(pred_df.style.highlight_max(axis=1, color='grey'))
				st.info(outcomes[np.int(test_class)])
			

			if test_class == choice['label']:
				st.success("Prediction Correct")
			else:
				st.error("Prediction Incorrect. The correct prediction is " + outcomes[choice['label']])


if __name__ == '__main__':
	main()

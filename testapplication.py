#!/usr/bin/env python3
import subprocess
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.vectorstores import Chroma
from langchain.llms import GPT4All, LlamaCpp
import chromadb
import os
import argparse
import time
from flask import Flask, render_template, request, jsonify, session, redirect, url_for
import hashlib

if not load_dotenv():
    print("Could not load .env file or it is empty. Please check if it exists and is readable.")
    exit(1)

embeddings_model_name = os.environ.get("EMBEDDINGS_MODEL_NAME")
persist_directory = os.environ.get('PERSIST_DIRECTORY')

model_type = os.environ.get('MODEL_TYPE')
model_path = os.environ.get('MODEL_PATH')
model_n_ctx = os.environ.get('MODEL_N_CTX')
model_n_batch = int(os.environ.get('MODEL_N_BATCH',8))
target_source_chunks = int(os.environ.get('TARGET_SOURCE_CHUNKS',1))

from constants import CHROMA_SETTINGS

app= Flask(__name__)
app.secret_key= "secret key"


def process_input(query):

    embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)
    chroma_client = chromadb.PersistentClient(settings=CHROMA_SETTINGS , path=persist_directory)
    db = Chroma(persist_directory=persist_directory, embedding_function=embeddings, client_settings=CHROMA_SETTINGS, client=chroma_client)
    retriever = db.as_retriever(search_kwargs={"k": target_source_chunks})

    callbacks = [StreamingStdOutCallbackHandler()]

    match model_type:
        case "LlamaCpp":
            llm = LlamaCpp(model_path=model_path, max_tokens=model_n_ctx, n_batch=model_n_batch, callbacks=callbacks, verbose=False)
        case "GPT4All":
            llm = GPT4All(model=model_path, max_tokens=model_n_ctx, backend='gptj', n_batch=model_n_batch, callbacks=callbacks, verbose=False)
        case _default:
            # raise exception if model_type is not supported
            raise Exception(f"Model type {model_type} is not supported. Please choose one of the following: LlamaCpp, GPT4All")

    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents= True)

    start = time.time()
    res = qa(query)
    answer, docs = res['result'], res['source_documents']
    end = time.time()

    # response= f"Question: {query} \n Answer: {answer} \n Time: {round(end - start, 2)} s. \n Relevant sources: {docs}"
    response= f"Question: {query} <br> <br> Answer: {answer} <br> <br> Time: {round(end - start, 2)} s."

    return response


@app.route('/')
def home():
    if 'username' in session:
        return render_template('index.html')
    else:
        return redirect(url_for('login'))


@app.route('/process', methods=['POST'])
def process():
    if 'username' not in session:
        return redirect(url_for('login'))

    query= request.form['query']
    result= process_input(query)

    # return render_template('result.html', result= result)
    return jsonify({'result': result})



# Define a route to run ingest.py
@app.route('/run_ingest')
def run_ingest():
    if 'username' not in session:
        return redirect(url_for('login'))

    try:
        subprocess.run(['python', 'ingest.py'], check=True, text=True)
        return 'ingest.py executed successfully'
    except subprocess.CalledProcessError as e:
        return f'Error executing ingest.py: {e}', 500
    
@app.route('/get_file_list')
def get_file_list():
    if 'username' not in session:
        return redirect(url_for('login'))

    directory_path = '/Users/jyotit-kaushal/github/Private-Chatbot/source_documents'
    
    try:
        file_list = [f for f in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, f))]
        file_list_html = ''.join([f'<li><input type="checkbox" name="files" value="{file}">{file}</li>' for file in file_list])
        return file_list_html
    except Exception as e:
        return f'Error getting file list: {e}', 500

# Define a route to delete selected files
@app.route('/delete_files', methods=['POST'])
def delete_files():
    if 'username' not in session:
        return redirect(url_for('login'))

    try:
        data = request.get_json()
        files_to_delete = data.get('files', [])

        directory_path = '/Users/jyotit-kaushal/github/Private-Chatbot/source_documents'
        for file_name in files_to_delete:
            file_path = os.path.join(directory_path, file_name)
            os.remove(file_path)

        return 'Files deleted successfully'
    except Exception as e:
        return f'Error deleting files: {e}', 500
    

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')

        # Verify credentials from the users.txt file
        if verify_credentials(username, password):
            session['username'] = username
            return redirect(url_for('home'))
        else:
            return render_template('login.html', error='Invalid username or password.')

    return render_template('login.html')


@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect(url_for('login'))


def verify_credentials(username, password):
    hashed_password = hashlib.sha256(password.encode()).hexdigest()

    # Check against the users.txt file (replace with a secure database in production)
    users_file_path = 'users.txt'
    with open(users_file_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            stored_username, stored_hashed_password = line.strip().split(':')
            if username == stored_username and hashed_password == stored_hashed_password:
                return True
    return False


@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')

        # Check if the username is already taken
        if user_exists(username):
            return render_template('signup.html', error='Username already exists. Please choose another username.')

        # Hash the password before storing it
        hashed_password = hashlib.sha256(password.encode()).hexdigest()

        # Save the new user to the users.txt file
        with open('users.txt', 'a') as file:
            file.write(f'{username}:{hashed_password}\n')

        return redirect(url_for('login'))

    return render_template('signup.html')

def user_exists(username):
    users_file_path = 'users.txt'
    with open(users_file_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            stored_username, _ = line.strip().split(':')
            if username == stored_username:
                return True
    return False

if __name__ == "__main__":
    app.run(debug=True)

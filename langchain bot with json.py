from flask import Flask, flash, request, redirect, url_for, jsonify, send_file, make_response,session
from flask_cors import CORS, cross_origin
import json
from langchain.agents import create_json_agent
from langchain.agents.agent_toolkits import JsonToolkit
from langchain.tools.json.tool import  JsonSpec
from langchain.chat_models import ChatOpenAI
from langchain.llms import GooglePalm
from langchain_openai import OpenAI
from langchain_core.prompts import PromptTemplate
import os

os.environ["OPENAI_API_KEY"] = ""

with open("experiences_update.json", "r") as f:
    experience_data = json.load(f)

app = Flask(__name__)
CORS(app, support_credentials=True)
app.config['CORS_HEADERS'] = 'Content-Type'

app.secret_key = 'my_secret_key'

@app.route('/', methods=['GET', 'POST'])
@cross_origin(supports_credentials=True)
def hello():
    return jsonify("Hello World!")

@app.route('/get_result', methods=['POST'])
def get_result():
    response = request.get_json()
    customer_query = response["customerQuery"]
    previous_chat = response["previousChat"]
    customer_data = response["customerData"]

    if len(previous_chat) == 0:
        llm = OpenAI(model_name="gpt-3.5")
        question = f"The question of user is {customer_query}, now ask the customer about his prefernce according to question. make sure it should be approprite according to the question and small."
        answer = llm.invoke(question)
        if "\n" in answer:
            answer = answer.split("\n")[-1]
        return jsonify({"Output": answer})
    elif len(previous_chat) == 1:
        llm = OpenAI(model_name="gpt-3.5")
        question = f"""
        Question asked from the user: {previous_chat[0]["content"]}
        The answer of user is {customer_query}, now ask the customer about the specifics according to preference he told. make sure it should be approprite according to the preference and small."""
        answer = llm.invoke(question)
        if "\n" in answer:
            answer = answer.split("\n")[-1]
        return jsonify({"Output": answer})
    
    else:
        with open("experiences_update.json", "r") as f:
            experience_data = json.load(f)
        data = {**customer_data, **experience_data}
        spec=JsonSpec(dict_=data,max_value_length=4000)
        toolkit=JsonToolkit(spec=spec)
        agent=create_json_agent(llm=ChatOpenAI(temperature=0,model="gpt-3.5-turbo"),toolkit=toolkit,max_iterations=1000,verbose=True,handle_parsing_errors=True)
        answer = agent.run(f"""
        Question 1: {previous_chat[0]["content"]}
        Answer: the preference of the user is {previous_chat[1]["role"]}
        Question 2: {previous_chat[1]["content"]}
        Answer: {customer_query}
        
        Now based on the customer answers above, answer the customer query as an expert travel agent and also keep in mind the customer deatils. customer query you have to reply is: {previous_chat[0]["content"]}. Don't mention json and just provide the most relevant experience ids according to the customer details. If you don't find any relevant experience then don't write anything, if you find then write the most relevant experienceIds(ids/numeric) only.
        Remember to check the customer details and experiences from json and based on that provide the ids of the suitable places. There should be some ids.
        """)
        if "I don't know" in answer:
            return {"output": answer}
        else:
            return {"output": answer.split(",")}
        
# main driver function
if __name__ == '__main__':
 
    # run() method of Flask class runs the application 
    # on the local development server.
    app.run(host='0.0.0.0', port=8078)
# import streamlit as st
# from streamlit_chat import message
# from streamlit_extras.colored_header import colored_header
# from streamlit_extras.add_vertical_space import add_vertical_space
from langchain import PromptTemplate, HuggingFaceHub, LLMChain
from classifier.moodclassificationspotify import predict_emotion, max_len
from classifier.genre_classifyspotify import *
from feature_extract.feature_genre import *
from feature_extract.main import *
from PIL import Image
import io
import requests
import matplotlib.pyplot as plt
from EmotionFacialRecog.emotion_recog import *

API_URL = "https://api-inference.huggingface.co/models/SG161222/Realistic_Vision_V1.4"
headers = {"Authorization": "Bearer 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"}


def query(payload):
	response = requests.post(API_URL, headers=headers, json=payload)
	return response.content

from dotenv import load_dotenv

# load the Environment Variables. 
load_dotenv()


def prompt_generate(predicted_emotion, predicted_genre, predicted_facial):
    prompt = f"Give a 3-5 liner prompt to generate an art of a person who is {predicted_facial} to hear a {predicted_genre} music giving {predicted_emotion} emotion that will make the viewer {predicted_emotion}."
    return prompt

def chain_setup():
    template = """<|prompter|>{question}<|endoftext|>
    <|assistant|>"""
        
    prompt = PromptTemplate(template=template, input_variables=["question"])

    llm=HuggingFaceHub(repo_id="OpenAssistant/oasst-sft-4-pythia-12b-epoch-3.5", model_kwargs={"max_new_tokens":1200})

    llm_chain=LLMChain(
    llm=llm,
    prompt=prompt
    )
    return llm_chain


    # generate response
def generate_response(question, llm_chain):
    response = llm_chain.run(question)
        # print(response)
    return response

    ## load LLM
llm_chain = chain_setup()

    
token = get_token()
track_id = get_track_id('Jimmy Cooks', token)

features_1 = get_features(track_id, token)

features_1['length'] = (features_1['length'])/max_len

features2 = get_feature(track_id, token)

file = r"C:\Users\tiwar\Artgenerator\Chat-App-OpenAssistant-API\EmotionFacialRecog\happy_img.jpeg"
pred_genre = predict_genre(features2)
pred_emotion = predict_emotion(features_1)
pred_facial = emotion_recog(file)
prompt = prompt_generate(pred_emotion, pred_genre, pred_facial)

response = generate_response(prompt, llm_chain)
print(response)
image_bytes = query({
	"inputs": response
})
# You can access the image with PIL.Image for example
image = Image.open(io.BytesIO(image_bytes), formats=['JPEG'])
plt.imshow(image)
plt.show()




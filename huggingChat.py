import streamlit as st
from streamlit_chat import message
from streamlit_extras.colored_header import colored_header
from streamlit_extras.add_vertical_space import add_vertical_space
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
# st.set_page_config(page_title="OpenAssistant Powered Chat App")

# # Sidebar contents
# with st.sidebar:
#     st.title('🤗💬 HuggingChat App')
#     st.markdown('''
#     ## About
#     This app is an LLM-powered chatbot built using:
#     - [Streamlit](https://streamlit.io/)
#     - [LangChain](https://python.langchain.com/)
#     - [OpenAssistant/oasst-sft-4-pythia-12b-epoch-3.5](https://huggingface.co/OpenAssistant/oasst-sft-4-pythia-12b-epoch-3.5) LLM model

#     ''')
#     add_vertical_space(3)
#     st.write('Made with ❤️ by [Prompt Engineer](https://youtube.com/@engineerprompt)')

# st.header("Your Personal Assistant 💬")

# def main():

    # # Generate empty lists for generated and user.
    # ## Assistant Response
    # if 'generated' not in st.session_state:
    #     st.session_state['generated'] = ["I'm Assistant, How may I help you?"]

    # ## user question
    # if 'user' not in st.session_state:
    #     st.session_state['user'] = ['Hi!']

    # # Layout of input/response containers
    # response_container = st.container()
    # colored_header(label='', description='', color_name='blue-30')
    # input_container = st.container()

    # # get user input
    # def get_text():
    #     input_text = st.text_input("You: ", "", key="input")
    #     return input_text

    # ## Applying the user input box
    # with input_container:
    #     user_input = get_text()

def prompt_generate(predicted_emotion, predicted_genre, predicted_facial):
    prompt = f"Give a 3-5 liner prompt to generate an art of a person who is {predicted_facial} to hear a {predicted_genre} music giving {predicted_emotion} emotion that will make the viewer {predicted_emotion}."
def prompt_generate(predicted_emotion, predicted_genre):
    prompt = f"Give a 3-5 liner prompt to generate an art for {predicted_genre} music giving {predicted_emotion} emotion that will make the viewer {predicted_emotion}."
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



    # main loop
    # with response_container:
    #     if user_input:
    #         response = generate_response(user_input, llm_chain)
    #         st.session_state.user.append(user_input)
    #         st.session_state.generated.append(response)
            
    #     if st.session_state['generated']:
    #         for i in range(len(st.session_state['generated'])):
    #             message(st.session_state['user'][i], is_user=True, key=str(i) + '_user')
    #             message(st.session_state["generated"][i], key=str(i))

# if __name__ == '__main__':
#     main()

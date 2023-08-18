import os

import replicate
import streamlit as st
from dotenv import load_dotenv
from elevenlabs import generate
from langchain import PromptTemplate
from langchain.chains import LLMChain
from langchain.llms import OpenAI


# Load environment variables
load_dotenv()

# Load OpenAI API key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# Load Eleven Labs API key
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")


# initialize OpenAI API
llm = OpenAI(temperature = 0.5 )

def generate_recipe(food, calories):
    # Create a prompt
    prompt = PromptTemplate(
        input_variables = ["food", "calories"],
        template = """
            You are an experienced chef. Craft a recipe for {food} containing {calories} calories. Avoid employing symbols like -, +, *, or similar in the recipe. Instead of periods, utilize "Step" followed by the appropriate number for ordered lists, such as Step 1, Step 2, and so on.
"""
    )
    # Create a chain
    llm_chain = LLMChain(llm=llm, prompt = prompt)

    # Generate a recipe
    recipe = llm_chain.run({
        "food": food,
        "calories": calories
    })

    return recipe

def generate_audio(recipe, voice):
    audio = generate(text = recipe, voice = voice, api_key=ELEVENLABS_API_KEY)
    return audio

def generate_image(food):
    output = replicate.run(
    "stability-ai/stable-diffusion:ac732df83cea7fff18b8472768c88ad041fa750ff7682a21affe81863cbe77e4",
    input={"prompt": food}
    )

    return output


def app():
    st.title("Recipe Generator")
    food = st.text_input("What do you want to cook?")
    calories = st.number_input("How many calories?")
    voice_options = ["Rachel", "Bella"]
    voice = st.selectbox("Who do you want to read the recipe?", voice_options)
    submit_button = st.button("Generate")

    if submit_button:
        recipe = generate_recipe(food, calories)
        st.markdown(recipe)

        st.audio(generate_audio(recipe, voice))
        

        # Generate image
        images = generate_image(food)
        # Display image
        st.image(images[0])
        
    
    

if __name__ == "__main__":
    app()

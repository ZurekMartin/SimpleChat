import torch
import streamlit as st
from transformers import T5Tokenizer, T5ForConditionalGeneration, GPT2Tokenizer, GPT2LMHeadModel


# This function initializes the model and tokenizer based on the model name
@st.cache(allow_output_mutation=True)
def initialize_model(model_name):
    if model_name == "T5":
        tokenizer = T5Tokenizer.from_pretrained("t5-large")
        model = T5ForConditionalGeneration.from_pretrained("t5-large")
    elif model_name == "GPT2":
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2-large")
        model = GPT2LMHeadModel.from_pretrained("gpt2-large")
    else:
        raise ValueError("Invalid model name")

    return tokenizer, model


# This function generates a response based on the input text, model, tokenizer, and max length
def generate_response(input_text, model, tokenizer, length):
    try:
        if isinstance(model, GPT2LMHeadModel):
            with torch.no_grad():
                input_ids = tokenizer.encode(input_text, return_tensors="pt")
                response_ids = model.generate(input_ids, max_length=length, num_return_sequences=1)
            response = tokenizer.decode(response_ids[0], skip_special_tokens=True)
        elif isinstance(model, T5ForConditionalGeneration):
            with torch.no_grad():
                input_ids = tokenizer(input_text, return_tensors="pt").input_ids
                response_ids = model.generate(input_ids, max_length=length)
            response = tokenizer.decode(response_ids[0], skip_special_tokens=True)
        else:
            raise ValueError("Model not recognized or supported.")

        # Truncate the response to the last sentence, everything after the last period is removed
        last_period_index = response.rfind('.')
        if last_period_index != -1:
            response = response[:last_period_index + 1]

    except Exception as e:
        response = f"An error occurred while generating the response: {str(e)}"

    return response


# This is the main function that runs the Streamlit app
def main():
    st.set_page_config(page_title="SimpleChat")

    st.markdown("<h1 style='text-align: center;'>SimpleChat</h1>", unsafe_allow_html=True)

    with st.sidebar:
        st.title("Settings")
        username = st.text_input("Username:", "User", key="username")
        default_model = st.selectbox("Default model:", ["T5", "GPT2"], index=1)
        default_max_length = st.selectbox("Default max length:", [32, 64, 86, 128], index=1)

    user_input = st.chat_input("Enter your message...")

    tokenizer, model = initialize_model(default_model)
    max_length = default_max_length

    # If the user input is not empty, display the user input and generate a response
    if user_input:
        # If the username is "User", display the user input as a chat message
        if username == "User":
            with st.chat_message("user"):
                st.write(user_input)
        # Otherwise write the message with the username as a text
        else:
            st.write(f"{username}: {user_input}")

        response = generate_response(user_input, model, tokenizer, max_length)

        with st.chat_message("ai"):
            st.write(response)


# If the script is run directly, run the main function
if __name__ == "__main__":
    main()

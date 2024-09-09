import streamlit as st
from PIL import Image
import openai
import time

# Set your OpenAI API key here
openai.api_key = "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"

# Set the page title
st.title("Test Scenario Description Tool")

# Case 1: Text box for optional context
context = st.text_area("Optional Context", placeholder="Enter any additional context here...")

# Case 2: Multi-image uploader for screenshots (required)
uploaded_files = st.file_uploader("Upload screenshots (required)", accept_multiple_files=True, type=["png", "jpg", "jpeg"])

# Function to generate detailed testing instructions using LLM with retry mechanism
def generate_test_instructions(context, images, retries=3):
    prompt = f"Generate detailed testing instructions based on the following context: {context}. The test cases should cover the provided screenshots."

    delay = 2  # Initial delay between retries

    for attempt in range(retries):
        try:
            # Log the attempt
            st.write(f"Attempting to connect to OpenAI API... (Attempt {attempt + 1}/{retries})")

            # Send the request to the OpenAI API
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that generates test instructions."},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=500,
                temperature=0.7,
                request_timeout=300  # Further increased timeout to 300 seconds
            )

            # Log successful attempt
            st.write("Connection to OpenAI API was successful.")

            # Return the test instructions
            return response.choices[0].message["content"].strip()

        except openai.error.Timeout as e:
            if attempt < retries - 1:
                st.warning(f"Timeout occurred. Retrying... (Attempt {attempt + 2}/{retries}) after {delay} seconds.")
                time.sleep(delay)
                delay *= 2  # Exponential backoff
            else:
                st.error(f"Request timed out after {retries} attempts: {e}")
                return None
        except openai.error.OpenAIError as e:
            st.error(f"An error occurred with OpenAI: {e}")
            return None

# Case 3: Button to describe testing instructions
if st.button("Describe Testing Instructions"):
    if len(uploaded_files) == 0:
        st.error("Please upload at least one screenshot to proceed.")
    else:
        st.success("Test instructions are being generated based on the inputs.")
        
        # Display the context if provided
        if context:
            st.write(f"**Context provided:** {context}")
        
        # Display uploaded screenshots
        st.write("**Uploaded screenshots:**")
        for file in uploaded_files:
            st.image(file, caption=file.name)

        # Call the function to generate test instructions with retry mechanism
        test_instructions = generate_test_instructions(context, uploaded_files)

        # Display the generated testing instructions
        if test_instructions:
            st.write("### Generated Testing Instructions:")
            st.write(test_instructions)

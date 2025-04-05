import streamlit as st 
import requests
import json
import dotenv
import os
dotenv.load_dotenv()

# Initialize session state variables
if 'analyzed' not in st.session_state:
    st.session_state.analyzed = False
if 'feedback_given' not in st.session_state:
    st.session_state.feedback_given = False
if 'sentiment' not in st.session_state:
    st.session_state.sentiment = ""
if 'sentence' not in st.session_state:
    st.session_state.sentence = ""
if 'probability' not in st.session_state:
    st.session_state.probability = 0

BACKEND_API = os.getenv("BACKEND_API", "http://localhost:8000")
st.title("Vietnamese Sentiment Analysis for Comments")
st.header("Model: TextCNN trained on NTC-SCV dataset")

# Function to submit feedback
def submit_feedback(is_correct):
    if is_correct:
        # If correct, use the predicted sentiment
        label = 1 if st.session_state.sentiment == "positive" else 0
    else:
        # If incorrect, flip the sentiment
        label = 0 if st.session_state.sentiment == "positive" else 1
    
    try:
        feedback_response = requests.post(
            f"{BACKEND_API}/api/addsample",
            json={"text": st.session_state.sentence, "label": label},
        )
        if feedback_response.status_code == 200:
            st.session_state.feedback_given = True
            st.success("Feedback saved successfully!")
        else:
            st.error(f"Failed to save feedback: {feedback_response.text}")
    except Exception as e:
        st.error(f"Error submitting feedback: {e}")

# Input form
sentence = st.text_input("Fill in a sentence in Vietnamese:")

# Analysis button
if st.button("Analyze Sentiment"):
    if sentence:
        try:
            st.session_state.sentence = sentence
            st.session_state.analyzed = True
            st.session_state.feedback_given = False
            
            # Call API
            response = requests.post(
                f"{BACKEND_API}/api/predict",
                json={"text": sentence},
            )
            
            if response.status_code == 200:
                data = response.json()
                st.session_state.sentiment = data["sentiment"]
                st.session_state.probability = data["probability"]
                
                # Display result
                if st.session_state.probability > 0.9:
                    st.success(f"Sentiment: {st.session_state.sentiment} with probability: {st.session_state.probability}")
                else:
                    st.info(f"Sentiment: {st.session_state.sentiment} with probability: {st.session_state.probability}")
            else:
                st.error(f"Error from backend: {response.text}")
                st.session_state.analyzed = False
        except Exception as e:
            st.error(f"Error: {e}")
            st.session_state.analyzed = False
    else:
        st.warning("Please enter a sentence to analyze.")

# Only show feedback options if analysis was completed and feedback hasn't been given yet
if st.session_state.analyzed and not st.session_state.feedback_given:
    st.write("Is this result correct? If not, please provide feedback")
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("👍 Yes, correct", key="btn_correct"):
            submit_feedback(True)
    
    with col2:
        if st.button("👎 No, incorrect", key="btn_incorrect"):
            submit_feedback(False)

# Show thank you message after feedback is given
if st.session_state.feedback_given:
    st.success("Thank you for your feedback! It helps improve our model.")
    if st.button("Analyze another sentence"):
        st.session_state.analyzed = False
        st.session_state.feedback_given = False
        st.rerun()
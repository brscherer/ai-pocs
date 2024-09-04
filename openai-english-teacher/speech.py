import streamlit as st
import speech_recognition as sr
import pygame
from dotenv import load_dotenv
from pathlib import Path
from openai import OpenAI

load_dotenv()

client = OpenAI()

def call_open_api(message):
    completion = client.chat.completions.create(
        model='gpt-4o-mini',
        messages=[
            {"role": "system", "content": "I want you to act as a spoken English teacher and improver. I will speak to you in English, and you will reply to me in English to practice my spoken English. Keep your replies neat, limiting them to 100 words. Strictly correct my grammar mistakes, typos, and factual errors. Mark corrections with strikethroughs for removed words and **bold** for added words. Also, give me a score out of 100 for my responses. Please ask me a question in your reply. Remember to keep the feedback constructive and helpful."},
            {'role': 'user', 'content': message}
        ],
        temperature=0.7,
        stream=True
    )
    return completion

def collect_response_chunks(response_stream):
    collected_chunks = []
    collected_messages = []
    for chunk in response_stream:
        collected_chunks.append(chunk)
        chunk_message = chunk.choices[0].delta.content
        if chunk_message:
            collected_messages.append(chunk_message)
            if '.' in chunk_message:
                break  # Stop collecting when a full stop is found

    return ''.join(m for m in collected_messages if m)

def process_speech(text):
    response_stream = call_open_api(text)
    return collect_response_chunks(response_stream)

def main():
    pygame.mixer.init()
    st.title('Your English Teacher')

    if st.button("Start Speech Recognition"):
        recognizer = sr.Recognizer()
        with sr.Microphone() as source:
            st.write("Listening...")
            recognizer.adjust_for_ambient_noise(source)
            audio = recognizer.listen(source)
            st.write("Processing...")
        
        try:
            text = recognizer.recognize_google(audio)
            st.write("You said:", text)
            response_text = process_speech(text)

            speech_file_path = Path(__file__).parent / "speech.mp3"
            with client.audio.speech.with_streaming_response.create(
                model="tts-1",
                voice="alloy",
                input=response_text
            ) as response:
                response.stream_to_file(speech_file_path)
            st.write("Your Assistant:")
            st.write(response_text)
            
            pygame.mixer.music.load(speech_file_path)
            pygame.mixer.music.play()

        except sr.UnknownValueError:
            st.write("Sorry, could not understand audio.")
        except sr.RequestError as e:
            st.write(f"Could not request results; {e}")

if __name__ == "__main__":
    main()

import streamlit as st
from PIL import Image
from SimpsonsPredict import load_model, prediction

model, transforms=load_model('best_model.pth', 42)

st.title("Simpsons Facial Recognition App")
st.sidebar.markdown("Try Out the App!")
st.subheader("Instructions:")
st.write("Hello! This is an app designed to identify characters from the Simpsons TV show. Upload your images below! Please note that even if images contain multiple characters, the app will only return one label.")

files=st.file_uploader(label='Upload your images here', type=['png', 'jpg', 'jpeg'], accept_multiple_files=True)

for file in files:
    img=Image.open(file)
    preds=prediction(img, model, transforms)

    st.image(file)
    st.write(preds)
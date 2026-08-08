import streamlit as st

st.title("Training Process")
st.sidebar.markdown("About the Training Process")

st.write("This app uses an EfficientNetB0 model finetuned on a Simpsons character dataset (https://www.kaggle.com/datasets/alexattia/the-simpsons-characters-dataset/data).")
st.write("This model was trained in a Google Colab notebook on a T4 GPU.")



st.subheader("Importing the dataset")
with open("code_fragments/importing.txt", 'r') as f:
    lines=f.read()
    st.code(lines)

st.subheader("Data transformations")
with open("code_fragments/transformations.txt", 'r') as f:
    lines=f.read()
    st.code(lines)

st.subheader("Loading dataset")
with open("code_fragments/loading.txt", 'r') as f:
    lines=f.read()
    st.code(lines)

st.subheader("Building model")
with open("code_fragments/building.txt", 'r') as f:
    lines=f.read()
    st.code(lines)

st.subheader("Training and validation steps")
with open("code_fragments/training_step.txt", 'r') as f:
    lines=f.read()
    st.code(lines)
with open("code_fragments/valid_step.txt", 'r') as f:
    lines=f.read()
    st.code(lines)

st.subheader("Hyperparameters")
with open("code_fragments/hyper.txt", 'r') as f:
    lines=f.read()
    st.code(lines)

st.subheader("Setup")
with open("code_fragments/setup.txt", 'r') as f:
    lines=f.read()
    st.code(lines)

st.subheader("Training")
with open("code_fragments/training.txt", 'r') as f:
    lines=f.read()
    st.code(lines)
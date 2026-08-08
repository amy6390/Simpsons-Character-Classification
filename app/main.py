import streamlit as st

app=st.Page('app.py', title="Try Out the App!")
about=st.Page('about.py', title="About the Training Process")

nav=st.navigation([app, about])
nav.run()


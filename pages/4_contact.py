import streamlit as st

st.title('Contact Me')
st.write('Please fill out the form below to contact me:')
name = st.text_input('Name')
email = st.text_input('Email')
message = st.text_area('Message')
if st.button('Submit'):
    # Code to send email
    st.write('Thank you for contacting me!')

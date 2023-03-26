
from PIL import Image
import streamlit as st
import os
from datetime import datetime

st.set_page_config(page_title='Tobias Contact', page_icon="🎓",
                   initial_sidebar_state="expanded")  # layout='wide'

st.title('Get in contact with me😊')
# st.write('Please fill out the form below to contact me:')
# name = st.text_input('Name')
# email = st.text_input('Email')
# message = st.text_area('Message')
# if st.button('Submit'):
#     try:
#         assert name and email and message

#         # Create the file if it doesn't exist
#         filename = 'contact_info.txt'
#         if not os.path.exists(filename):
#             with open(filename, 'w') as f:
#                 pass

#         # Write the contact information to the file
#         with open(filename, 'a') as f:
#             now = datetime.now()
#             timestamp = now.strftime("%Y-%m-%d %H:%M:%S")
#             f.write(
#                 f'Timestamp: {timestamp}\nName: {name}\nEmail: {email}\nMessage: {message}\n\n')

#         # Code to send email
#         st.write('Thank you for contacting me! 😊')
#     except:
#         st.warning('Please fill in your contact details and a message', icon="⚠️")


def clickable_card(image_url: str, redirect_url: str):
    # Load image and create card
    image = Image.open(image_url)
    st.image(image, width=300, use_column_width=True)
    card = st.empty()
    card.markdown(f'<div style="border-radius: 10px; padding: 10px; background-color: white;">'
                  f'<a href="{redirect_url}" target="_blank">'
                  f'<h3 style="margin: 0;">Click Here</h3></a></div>', unsafe_allow_html=True)


col1, col2 = st.columns(2)
with col1:
    clickable_card("references/linkedin.png",
                   'https://www.linkedin.com/in/tobias-fechner-/')
with col2:
    clickable_card("references/GitHub2.jpeg", 'https://github.com/sceeptex')

import streamlit as st
import os
from datetime import datetime

st.set_page_config(page_icon="🎓",
                   initial_sidebar_state="expanded")  # layout='wide'


st.title('Contact Me')
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

# Footer
# st.markdown("---")
# st.header("Let's connect!")
link = '[LinkedIn](https://www.linkedin.com/in/tobias-fechner-/)'
st.markdown(link, unsafe_allow_html=True)
link = "[Github](https://github.com/sceeptex)"
st.markdown(link, unsafe_allow_html=True)

import streamlit as st
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Define your skills
skills = ['Python', 'Streamlit', 'Data Visualization',
          'Machine Learning', 'Deep Learning', 'Natural Language Processing']

# Create a string of skills for the word cloud
skills_str = ' '.join(skills)

# Generate the word cloud
wordcloud = WordCloud(background_color='white', width=800, colormap='coolwarm',
                      height=400).generate(skills_str)

# Display the word cloud
fig, ax = plt.subplots(figsize=(15, 7))
ax.imshow(wordcloud, interpolation='bilinear')
ax.axis('off')

# Define the layout of your website
st.set_page_config(page_title="My Streamlit Portfolio",
                   page_icon=":guardsman:", layout="wide")
st.title("Welcome to My Streamlit Portfolio")
st.write("Here's a word cloud showing my skills:")

st.pyplot(fig)

st.write("I am proficient in using Streamlit for building interactive web applications. If you're a recruiter looking for someone with these skills, please don't hesitate to reach out!")

st.write('Here are some of the projects that I have worked on:')


# Add your projects and descriptions here
st.write('- Project 1')
st.write('- Project 2')
st.write('- Project 3')

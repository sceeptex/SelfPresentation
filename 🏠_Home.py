import streamlit as st
from wordcloud import WordCloud
import matplotlib.pyplot as plt

st.set_page_config(page_title='Home', page_icon="🎓",
                   initial_sidebar_state="expanded")  # layout='wide'


st.title('Welcome to My Streamlit Portfolio!')
st.write('Hi, I am [Your Name], and I am a Streamlit developer.')
st.write('I specialize in creating custom Streamlit apps that help organizations and individuals to make data-driven decisions. ')
st.write('Feel free to explore my portfolio and contact me if you have any questions or projects in mind!')


def create_wordcloud(skills):
    # Create a string of skills for the word cloud
    skills_str = ' '.join(skills)

    # Generate the word cloud
    wordcloud = WordCloud(background_color=None, mode="RGBA", width=800,  colormap='viridis',
                          height=400).generate(skills_str)

    # Display the word cloud
    fig, ax = plt.subplots(figsize=(15, 7))  # , facecolor='k')
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    return fig


st.write("Here's a word cloud showing my skills:")

tab1, tab2, tab3 = st.tabs(["Cat", "Dog", "Owl"])

with tab1:
    skills = ['Python', 'Streamlit', 'Data Visualization',
              'Machine Learning', 'Deep Learning', 'Natural Language Processing']
    fig = create_wordcloud(skills)
st.pyplot(fig)

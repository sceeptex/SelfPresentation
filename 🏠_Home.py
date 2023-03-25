from PIL import Image
import streamlit.components.v1 as components
import streamlit as st
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# embed_component = {'linkedin': """<script src="https://platform.linkedin.com/badges/js/profile.js" async defer type="text/javascript"></script>
# <div class="badge-base LI-profile-badge" data-locale="en_US" data-size="large" data-theme="light" data-type="VERTICAL" data-vanity="tobias-fechner-" data-version="v1"><a class="badge-base__link LI-simple-link" href="https://de.linkedin.com/in/tobias-fechner-?trk=profile-badge">Tobias Fechner</a></div>
#               """
#                    }

st.set_page_config(page_title='Tobias Portfolio', page_icon="🎓",
                   initial_sidebar_state="expanded")  # layout='wide'


st.title('Welcome Everyone 😊')
st.write('My name is Tobias and I am an aspiring Data Science student with 4+ years of experience in IT and AI/ML Research who is passionate about building Machine Learning systems that have a real-world impact. I have strong technical skills, especially in Python, Machine Learning, Deep Learning, and databases, as well as an academic background in mathematics, programming, business and statistics.')
st.write('My passion is solving business challenges with unique approaches and communicating complex ideas to business stakeholders.')
st.write('Feel free to connect with me and to explore my website.')


# def create_wordcloud(skills):
#     # Create a string of skills for the word cloud
#     skills_str = ' '.join(skills)

#     # Generate the word cloud
#     wordcloud = WordCloud(background_color=None, mode="RGBA", width=800,  colormap='viridis',
#                           height=400).generate(skills_str)

#     # Display the word cloud
#     fig, ax = plt.subplots(figsize=(15, 7))  # , facecolor='k')
#     ax.imshow(wordcloud, interpolation='bilinear')
#     ax.axis('off')
#     return fig


st.header("Explore my Skills")


sections = {
    "Data Science": ["Statistics", "Machine_Learning", "Deep_Learning", "Natural_Language_Processing", "Computer_Vision", "Time_Series"],
    "Data Analysis": ["EDA", "Data_Visualization", "Matplotlib", "Plotly", "NumPy", "Pandas"],
    "Web Development": ["Python Backend", "Python Frontend", "Flask REST API", "Dash", "Streamlit", "HTML5", "CSS", "React Native"],
    "Tools & Frameworks": ["Git", "SQL", "NoSQL", "Neo4J", "Jupyter_Notebooks", "SciKit-learn", "TensorFlow", "Keras", "PyTorch", "PyTorch_Lightning", "Vision_Transformers", "BayesianSearch", "Optuna", "TensorBoard", "MLflow", "Lime", "SHAP", "Selenium", "Beautifulsoup", "Microsoft_PowerBI", "Microsoft_PowerAutomate", "LaTeX"],
    "Project Management": ["SCRUM", "BPMN", "UML"],
    "Cloud Computing": ["Google_Cloud_Platform", "Docker"],
}
# st.write(list(sections.keys()))
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(list(sections.keys()))

with tab1:
    st.image('references/wordcloud/Data_Science.jpg')
    # skills = sections['Data Science']
    # fig = create_wordcloud(skills)
    # st.pyplot(fig)
with tab2:
    st.image('references/wordcloud/Data_Analysis.jpg')
    # skills = sections['Data Analysis']
    # fig = create_wordcloud(skills)
    # st.pyplot(fig)
with tab3:
    st.image('references/wordcloud/Web_Development.jpg')
    # skills = sections['Web Development']
    # fig = create_wordcloud(skills)
    # st.pyplot(fig)
with tab4:
    st.image('references/wordcloud/Tools_Frameworks.jpg')
    # skills = sections['Tools & Frameworks']
    # fig = create_wordcloud(skills)
    # st.pyplot(fig)
with tab5:
    st.image('references/wordcloud/Project_Management.jpg')
    # skills = sections['Project Management']
    # fig = create_wordcloud(skills)
    # st.pyplot(fig)
with tab6:
    st.image('references/wordcloud/Cloud_Computing.jpg')
    # skills = sections['Cloud Computing']
    # fig = create_wordcloud(skills)
    # st.pyplot(fig)


# Footer
st.markdown("---")
st.header("Let's connect!")


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


# link = '[LinkedIn](https://www.linkedin.com/in/tobias-fechner-/)'
# st.markdown(link, unsafe_allow_html=True)
# link = "[Github](https://github.com/sceeptex)"
# st.markdown(link, unsafe_allow_html=True)

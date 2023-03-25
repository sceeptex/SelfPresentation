from PIL import Image
import streamlit as st


st.set_page_config(page_title='Tobias Exploratory Project Portfolio', page_icon="🎓",
                   initial_sidebar_state="expanded")  # layout='wide'


st.title('Project Portfolio')

st.subheader(
    'Power BI Dashboard for Electric Chargers: Increasing Service Technician Efficiency and Customer Satisfaction')
st.write('Jan 2022 - Feb 2022')

st.image('references/portfolio/powerBi.jpg',
         caption="Sample image from https://powerbi.microsoft.com/en-us/desktop/")
st.write("""The development of the electric charger Dashboard in Power BI is a major step forward for ABB service technicians. With this powerful tool, technicians are able to gain valuable insights into the performance and usage of each EV charger, helping them to provide the best possible service to customers.

Previously, technicians faced challenges in quickly identifying and resolving problems with EV chargers. They had to rely on manual methods to gather data and diagnose issues, which was time-consuming and error-prone. With the electric charger Dashboard in Power BI, this is no longer the case.

The dashboard provides real-time data on problems, usage patterns, and malfunctioning components, enabling technicians to quickly and accurately identify and fix issues. In addition, the dashboard can be used to perform predictive maintenance, which can prevent potential problems from arising in the first place. This is particularly useful in ensuring the reliability and performance of EV chargers, which is essential for providing top-quality service to customers.

Overall, the electric charger Dashboard in Power BI is a game-changing tool for ABB service technicians. It enables them to work more efficiently and effectively, helping them to provide the best possible service to customers.""")

st.subheader("Predicting Rent Prices in Mannheim: A Case Study with XGBoost")
st.write("Sep 2021 - Dec 2021")
col1, col2 = st.columns(2)
with col1:
    st.image(r'references\portfolio\rent0.png')
    st.image(r'references\portfolio\rent2.png')
    st.image(r'references\portfolio\rent4.png')
with col2:
    st.image(r'references\portfolio\rent1.png')
    st.image(r'references\portfolio\rent3.png')
    st.image(r'references\portfolio\rent5.png')
with st.expander('Learn more:', True):
    st.write("""
    In this university project, we used regression models and the XGBoost algorithm to predict rent prices for a dataset of rental properties in Germany. We collected and cleaned the dataset, trained a machine learning model using XGBoost, and evaluated its performance. We also compared our results to other regression methods and found that our model was able to provide more accurate and reliable predictions. Furthermore, we optimized the model hyperparameters with a Bayesian Search algorithm and achieved even better results. In addition to our technical findings, we also made recommendations for our peers on the best locations and types of apartments to consider when looking for a flat in Mannheim. This project demonstrates the effectiveness of XGBoost for rent price prediction and provides valuable insights into the factors that influence rent prices in Mannheim""")

st.subheader(
    "Object detection with Detectron2: The State-of-the-art model from Facebook for object detection")
st.write("Sep 2021 - Oct 2021")
st.image('references/portfolio/detetectron2.gif',
         caption="https://github.com/facebookresearch/detectron2")
with st.expander('Learn more:', True):
    st.write("""In this project, we used Detectron2, a powerful object detection framework, to train machine learning models for object detection. By leveraging the latest advancements in deep learning, we were able to achieve impressive performance on our object detection tasks. Our project demonstrates the capabilities of Detectron2 and its potential for real-world applications in fields such as process automation and robotics. Overall, this project showcases the effectiveness of modern deep learning techniques for object detection.""")

st.subheader(
    "Image Classification with CNNs: Accurate and Interpretable Predictions Using Explainable AI")
st.write("Oct 2020 - Oct 2020")
with st.expander('Learn more:', False):
    st.write("""In this project, we utilized convolutional neural networks (CNNs) for image classification. We developed our own CNNs as well as using pretrained models in combination with transfer learning from Keras. Our goal was not only to achieve high performance on our predictions, but also to gain a deeper understanding of CNNs through the use of explainable artificial intelligence libraries such as LIME, SHAP. We tracked our trainings with MLFlow and compared our results. By combining advanced machine learning techniques with interpretability tools, we were able to achieve a better understanding of our models and improve their performance. This project is a valuable demonstration of the potential of CNNs and explainable AI for image classification to provide more trust in deep learning models.""")

st.subheader("Building Machine Learning Models with Scikit-learn and Visualizing data with Plotly and Dash: A Comprehensive Dashboard for Data Scientists")
st.write("Sep 2020 - Sep 2020")
with st.expander('Learn more:', False):
    st.write("""In this project, I created an easy-to-use dashboard with Dash and Plotly that allows Data Scientists to gain insights into their data and discover correlations. With this tool, data scientists can visualize the raw data and build better machine learning models. The dashboard also provides a clear and concise way to communicate the results of these models, including detailed statistics and explanations of the model training, validation, and testing. Overall, this dashboard is a valuable tool for data scientists looking to gain a better understanding of their data and to build more effective machine learning models.""")

st.subheader(
    "Exploring the Capabilities of Motion Sensors with Machine Learning: A Study at the ABB Research Center")
st.write("Aug 2020 - Aug 2020")
with st.expander('Learn more:', False):
    st.write("In this project, we intended to explore the potential of motion sensors by using machine learning to interpret the time series data. This was my first project with Python and I learned a lot about machine learning and signal processing. Our goal was to develop what we call soft sensors, which are sensors that can be easily attached and use advanced computing techniques to extract information from data that is otherwise difficult or impossible to measure. By harnessing the power of machine learning and data processing techniques, we aimed to develop a new approach that would provide more accurate and comprehensive insights into physical phenomena. Through our research, we have gained a deeper understanding of the capabilities of motion sensors and how they can be used to develop advanced, intelligent sensing systems.")

st.subheader("Managing Transport Releases in SAP Systems: Enhancing Stability and Reliability through Improved Change Management")
st.write("Apr 2020 - Aug 2020")
with st.expander('Learn more:', False):
    st.write("""SAP systems have evolved significantly to include a wide variety of development objects and technologies. As the complexity and scope of these systems continue to grow, it becomes increasingly important to carefully monitor and manage changes to these objects to ensure smooth and seamless operation of the SAP system. In this project, we wanted to address this challenge by developing a system that automatically classifies changes based on a risk matrix. This risk matrix takes into account the probability of occurrence and impact that each development object can have. This involves checking in how many other developments the object is used and thus how high its dependency is, as well as analyzing the individual components of each development object. By implementing our system, we can gain a more detailed view of the changes that are being made to the SAP system and can take appropriate steps to ensure that these changes do not affect the stability, security, and performance of the SAP system.""")

st.subheader(
    "Assessing Readiness for Migration to SAP S/4HANA: A Comprehensive Overview of the Key Requirements")
st.write("Mar 2020 - Apr 2020")
with st.expander('Learn more:', False):
    st.write("""In this project, I implemented the SAP Readiness Check for SAP S/4HANA to get a comprehensive overview of the current status of the ERP landscape and to identify the most important aspects that need to be changed in order to migrate from the ERP system to a S/4HANA system. SAP S/4HANA is based on a high-performance in-memory database, providing a significant advancement in the field of enterprise resource planning. As such, it is essential for any organization operating in the ERP space to consider migrating to this platform in order to remain competitive and to stay at the forefront of technological innovation. The SAP Readiness Check is a valuable tool that can help organizations assess their readiness for a successful migration to SAP S/4HANA. By using this tool, organizations can gain a clear understanding of the key considerations and requirements involved in the migration process, enabling them to plan and implement their transition to S/4HANA efficiently.""")

st.subheader("Improving User Management in Satellite Systems: Enhancing Connectivity in the SAP Landscape to Increase Security")
st.write("Apr 2019 - May 2019")
with st.expander('Learn more:', False):
    st.write("""The goal of this project was to establish connectivity between the satellite systems and the main system in the SAP landscape for improved user management. This was done to ensure that each system had the up-to-date user credentials. With this new development, we improved the user management and were able to increase the security of the satellite systems.""")

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


import sys
import time
import pickle
import matplotlib as plt
from sklearn.exceptions import NotFittedError
from PIL import Image
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.datasets import load_iris
import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import numpy as np
from sklearn.tree import export_graphviz
import graphviz


st.set_page_config(page_title='Tobias Machine Learning', page_icon="🎓",
                   initial_sidebar_state="expanded")  # layout='wide'


np.random.seed(31421)

iris = load_iris()


@st.cache_data
def load_data():
    iris = load_iris()
    data = pd.DataFrame(iris.data, columns=iris.feature_names)
    data['target'] = iris.target
    data['species'] = data['target'].apply(lambda x: iris.target_names[x])
    return data


data = load_data()

y = data['species']
X = data.drop(['species', 'target'], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

# Create a Scikit-learn pipeline with a Random Forest model
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('rf', RandomForestClassifier())
])

# Create a Streamlit web application
st.title('Built your first Machine Learning Model')

# Create a form for the user to select hyperparameters
with st.form('hyperparameters'):
    st.write("## Random Forest Estimator")
    st.write("Change the hyperparameter:")
    # n_estimators = st.selectbox('Number of estimators', [
    #                             1, 2, 3, 5, 7, 10, 100, 200])
    # max_depth = st.selectbox('Maximum tree depth', [1, 2, 3, 5, 7, 9])
    col_1, col_2 = st.columns(2)
    with col_1:
        n_estimators = st.number_input(
            'Number of estimators', value=1, step=1)
    with col_2:
        max_depth = st.number_input(
            'Maximum tree depth', value=1, step=1)
    submit_button = st.form_submit_button(label='Train model')


@st.cache_resource
def train_model(X_train, y_train, n_estimators, max_depth):
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('rf', RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth))
    ])
    pipeline.fit(X_train, y_train)
    return pipeline


if 'accuracy_list' not in st.session_state:
    st.session_state.accuracy_list = []

# Train the model using the selected hyperparameters when the user clicks the submit button
if submit_button:
    with st.expander("Training Results", True):
        # Update the hyperparameters in the pipeline
        pipeline = train_model(X_train, y_train, n_estimators, max_depth)

        # Evaluate the model on the testing data
        accuracy = pipeline.score(X_test, y_test)

        # Show the results
        # st.write('Accuracy:', accuracy)
        st.session_state.accuracy_list.append(accuracy)

        # Compute the change in accuracy from the last run
        if len(st.session_state.accuracy_list) >= 2:
            delta = st.session_state.accuracy_list[-1] - \
                st.session_state.accuracy_list[-2]
        else:
            delta = 0.0

        # Show the results
        st.metric('Validation Accuracy', round(accuracy, 2), delta)

        y_pred = pipeline.predict(X_test)

        # Print the classification report
        # target_names = ['Setosa', "Versicolor", "Virginica"]
        # print(classification_report(y_test, y_pred, labels=target_names)
        # )

        st.subheader("Classification Report")
        report = classification_report(y_test, y_pred, target_names=[
            'Setosa', "Versicolor", "Virginica"])
        st.text(report)

        # Show confusion matrix
        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        cm_fig = go.Figure(data=go.Heatmap(z=cm, x=[0, 1, 2], y=[0, 1, 2], colorscale="Blues",
                                           hoverongaps=False))
        cm_fig.update_layout(title="Confusion Matrix",
                             xaxis_title="Predicted Label",
                             yaxis_title="True Label",
                             xaxis=dict(tickmode="array", tickvals=[
                                 0, 1, 2], ticktext=['Setosa', "Versicolor", "Virginica"]),
                             yaxis=dict(tickmode="array", tickvals=[0, 1, 2], ticktext=['Setosa', "Versicolor", "Virginica"]))
        st.plotly_chart(cm_fig)

        # show missclassifed examples
        test_df = X_test.copy()
        test_df['actual'] = y_test
        test_df['predicted'] = y_pred
        test_df['Wrong_Prediction'] = [a != p for a, p in zip(
            test_df['actual'], test_df['predicted'])]
        test_df['Wrong_Prediction'] = test_df['Wrong_Prediction'].replace(
            True, 'yes')
        test_df['Wrong_Prediction'] = test_df['Wrong_Prediction'].replace(
            False, 'no')

        fig = px.scatter_matrix(
            test_df,
            dimensions=['sepal length (cm)', 'sepal width (cm)',
                        'petal length (cm)', 'petal width (cm)'],
            color="Wrong_Prediction",
            title='Scatter plot matrix for Iris dataset',
            hover_data=['actual', 'predicted'],
            color_discrete_sequence=["#7fff00", "#FF0000"],
            opacity=0.7,
        )
        fig.update_traces(diagonal_visible=False)
        fig.update_layout(
            height=800)
        st.plotly_chart(fig)

with st.expander("Visualization of the Random Forest Model", False):
    # create the visual of the random forest
    pipeline = train_model(X_train, y_train, n_estimators, max_depth)
    for n_estimator in range(n_estimators):

        st.write("Tree number: ", n_estimator+1)
        tree = pipeline['rf'].estimators_[n_estimator]
        dot_data = export_graphviz(tree, out_file=None, feature_names=iris.feature_names,
                                   class_names=iris.target_names, filled=True, rounded=True,
                                   special_characters=True)
        graph = graphviz.Source(dot_data)

        # Create a streamlit expander to show the decision tree visualization
        st.graphviz_chart(graph.source)

with st.expander("Testing", True):
    # pipeline is cached
    pipeline = train_model(X_train, y_train, n_estimators, max_depth)

    st.markdown('### Test the Model with new data:')

    # sl = st.number_input('Sepal Length', value=5, step=1)
    sl = st.slider('Sepal Length', 0.1, 10.0, value=5.0)

    # sw = st.number_input('Sepal Width', value=3, step=1)
    sw = st.slider('Sepal Width', 0.1, 10.0, value=3.0)

    # pl = st.number_input('Petal Length', value=4, step=1)
    pl = st.slider('Petal Length', 0.1, 10.0, value=4.0)

    # pw = st.number_input('Petal Width', value=1, step=1)
    pw = st.slider('Petal Width', 0.1, 10.0, value=1.0)
    features = pd.DataFrame({'sepal length (cm)': [sl], 'sepal width (cm)': [
        sw], 'petal length (cm)': [pl], 'petal width (cm)': [pw]})

    prediction = pipeline.predict(features)[0]

    # st.write(f'The model predicts this flower is a **{prediction}**.')
    st.metric("The model predicts this flower is a:", prediction)

with st.expander("Save", True):
    st.markdown('### Save your model')

    # if st.button("Download model"):
    #     with open("iris_model.pkl", "wb") as f:
    #         pickle.dump(pipeline, f)
    #     st.success("Model downloaded successfully!")

    pipeline_bytes = pickle.dumps(pipeline)
    # Download the bytes as a file
    st.download_button(
        label="Download Model",
        data=pipeline_bytes,
        file_name="iris_model.pkl",
        mime="application/octet-stream",
    )

# Footer
st.markdown("---")
st.header("Let's connect!")
link = '[LinkedIn](https://www.linkedin.com/in/tobias-fechner-/)'
st.markdown(link, unsafe_allow_html=True)
link = "[Github](https://github.com/sceeptex)"
st.markdown(link, unsafe_allow_html=True)

import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import pickle
import plotly.express as px
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score


# def feature_engineering():
# st.sidebar.subheader("Feature Selection")
# data = load_data()
# target_variable = st.sidebar.selectbox(
#     "Select Target Variable", data.columns)
# feature_variables = st.sidebar.multiselect(
#     "Select Feature Variables", data.columns)
# selected_data = data[[target_variable] + feature_variables].dropna()

# st.sidebar.subheader("Feature Engineering")
# scaling = st.sidebar.checkbox("Scale Features?")
# one_hot_encoding = st.sidebar.checkbox("One-Hot Encoding?")
# if scaling:
#     selected_data = scale_features(selected_data, feature_variables)
# if one_hot_encoding:
#     selected_data = one_hot_encode(selected_data, feature_variables)

# st.session_state.selected_data = selected_data
# st.session_state.target_variable = target_variable

# def scale_features(data, features):
#     scaler = StandardScaler()
#     data[features] = scaler.fit_transform(data[features])
#     return data


# def one_hot_encode(data, features):
#     data = pd.get_dummies(data, columns=features)
#     return data


import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load data


@st.cache(allow_output_mutation=True)
def load_data():
    iris = load_iris()
    data = pd.DataFrame(iris.data, columns=iris.feature_names)
    data['target'] = iris.target
    data['species'] = data['target'].apply(lambda x: iris.target_names[x])
    return data

# Create feature engineering function


def create_new_feature(df, feature_1, feature_2, operation):
    if operation == "multiply":
        new_feature = df[feature_1] * df[feature_2]
    elif operation == "add":
        new_feature = df[feature_1] + df[feature_2]
    elif operation == "subtract":
        new_feature = df[feature_1] - df[feature_2]
    elif operation == "divide":
        new_feature = df[feature_1] / df[feature_2]
    return new_feature

# Create scatter plot function


def create_scatter_plot(df, x_col, y_col):
    fig = px.scatter(df, x=x_col, y=y_col, color='species')
    st.plotly_chart(fig)


# Create Streamlit app


def feature_engineering():
    # Load data
    data = load_data()

    # Scale data
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data.iloc[:, :-2])

    # Interactive feature engineering
    st.write("# Interactive Feature Engineering")
    feature_1 = st.selectbox("Select Feature 1", data.columns[:-2])
    feature_2 = st.selectbox("Select Feature 2", data.columns[:-2])
    operation = st.selectbox("Select Operation", [
                             "multiply", "add", "subtract", "divide"])

    if st.button("Create New Feature"):
        new_feature = create_new_feature(
            data_scaled, feature_1, feature_2, operation)
        st.write("New Feature:", new_feature)
        st.write("Correlation to Target:", round(
            pd.Series(new_feature).corr(data['target']), 2))
        data[f"{feature_1} {operation} {feature_2}"] = new_feature

    # Create scatter plot
    if 'new_feature' in data.columns:
        create_scatter_plot(data, 'new_feature', 'target')

    # Display data
    st.write("# Data with Created Features")
    st.write(data)


MODELS_DIR = "models/"


def save_model(model, name):
    """Save a trained model to disk."""
    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR)
    with open(os.path.join(MODELS_DIR, f"{name}.pkl"), "wb") as f:
        pickle.dump(model, f)


def load_model(name):
    """Load a trained model from disk."""
    with open(os.path.join(MODELS_DIR, f"{name}.pkl"), "rb") as f:
        model = pickle.load(f)
    return model


def model_development_and_training():
    st.sidebar.subheader("Model Selection")
    model_options = ["Logistic Regression", "Decision Tree", "Random Forest"]
    model_choice = st.sidebar.selectbox("Select Model", model_options)

    # Load selected data from Feature Engineering section
    selected_data = st.session_state.selected_data

    # Split the data into training and testing sets
    X = selected_data.drop(columns=[st.session_state.target_variable])
    y = selected_data[st.session_state.target_variable]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    st.subheader("Selected Data")
    st.write(selected_data.head())

    if model_choice == "Logistic Regression":
        # Select hyperparameters
        C = st.sidebar.slider(
            "C (Inverse of regularization strength)", 0.01, 10.0, 1.0)
        penalty = st.sidebar.selectbox("Penalty", ["none", "l2"])

        # Train the model
        model = LogisticRegression(C=C, penalty=penalty)
        model.fit(X_train, y_train)

    elif model_choice == "Random Forest":
        # Select hyperparameters
        n_estimators = st.sidebar.slider("Number of Estimators", 10, 100, 50)
        max_depth = st.sidebar.slider("Max Depth", 2, 20, 5)

        # Train the model
        model = RandomForestClassifier(
            n_estimators=n_estimators, max_depth=max_depth)
        model.fit(X_train, y_train)

    # Save the trained model if desired
    if st.sidebar.button("Save Model"):
        save_model(model)

    st.success("Model training complete")


def model_evaluation(model_file):
    st.title("Model Evaluation")

    # Load saved model
    model = load_model(model_file)

    # Load selected data
    data = pd.read_csv("data.csv")
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    # Make predictions on test data
    y_pred = model.predict(X_test)

    # Show classification report
    st.subheader("Classification Report")
    report = classification_report(y_test, y_pred)
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
                                    0, 1, 2], ticktext=["A", "B", "C"]),
                         yaxis=dict(tickmode="array", tickvals=[0, 1, 2], ticktext=["A", "B", "C"]))
    st.plotly_chart(cm_fig)

    # Show ROC curve
    st.subheader("ROC Curve")
    fpr, tpr, _ = roc_curve(
        y_test, model.predict_proba(X_test)[:, 1], pos_label=1)
    roc_auc = auc(fpr, tpr)
    roc_fig = go.Figure(data=go.Scatter(x=fpr, y=tpr, mode="lines",
                                        line=dict(color="navy", width=2),
                                        name=f"AUC={roc_auc:.2f}"))
    roc_fig.add_shape(type="line", x0=0, y0=0, x1=1, y1=1,
                      line=dict(color="gray", width=1, dash="dash"))
    roc_fig.update_layout(title="ROC Curve",
                          xaxis_title="False Positive Rate",
                          yaxis_title="True Positive Rate")
    st.plotly_chart(roc_fig)


st.title("ML Cycle Project")

# Sidebar Navigation
tab1, tab2, tab3 = st.tabs(["Feature Engineering",
                            "Model Development and Training", "Model Evaluation"])

with tab1:
    feature_engineering()
with tab2:
    model_development_and_training()
with tab3:
    model_evaluation()

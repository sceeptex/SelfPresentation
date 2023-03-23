from PIL import Image
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.datasets import load_iris

# Set app title
# Load dataset

st.set_page_config(page_title='Tobias Exploratory Data Analysis', page_icon="🎓",
                   initial_sidebar_state="expanded")  # layout='wide'


@st.cache_data
def load_data():
    iris = load_iris()
    data = pd.DataFrame(iris.data, columns=iris.feature_names)
    data['target'] = iris.target
    data['species'] = data['target'].apply(lambda x: iris.target_names[x])
    return data


data = load_data()

# Show dataset
# with st.expander("Exploratory Data Analysis (EDA)"):


image = Image.open('references/51518irisimg1.png')

st.title('The Iris Classification Challenge')
st.image(image, caption="The three classes of flowers")
st.markdown("""
The Iris flower data set or Fisher's Iris data set is a multivariate data set used and made famous by the British statistician and biologist Ronald Fisher in his 1936 paper The use of multiple measurements in taxonomic problems as an example of linear discriminant analysis. It is sometimes called Anderson's Iris data set because Edgar Anderson collected the data to quantify the morphologic variation of Iris flowers of three related species. 
Two of the three species were collected in the Gaspé Peninsula "all from the same pasture, and picked on the same day and measured at the same time by the same person with the same apparatus".

The data set consists of 50 samples from each of three species of Iris (Iris setosa, Iris virginica and Iris versicolor). Four features were measured from each sample: the length and the width of the sepals and petals, in centimeters. Based on the combination of these four features, Fisher developed a linear discriminant model to distinguish the species from each other. Fisher's paper was published in the Annals of Eugenics and includes discussion of the contained techniques' applications to the field of phrenology.
[[1]](https://en.wikipedia.org/wiki/Iris_flower_data_set)"

""")

st.write('## Dataset')
st.write(data.head())

# Show dataset shape
st.write('## Dataset Statistics')
st.write(data.describe())


# Show scatter plot
st.write('## Scatter Plot')
col1, col2, col3 = st.columns(3)
with col1:
    x_axis = st.selectbox('Select an x-axis column', data.columns[:-2])
with col2:
    y_axis = st.selectbox('Select a y-axis column', data.columns[:-2], index=2)
with col3:
    color = st.selectbox('Select a color column', ['species', 'target'])
fig = px.scatter(data, x=x_axis, y=y_axis, color=color,
                 title=f'{x_axis} vs {y_axis}', )
fig.update_traces(marker=dict(size=5))
# fig.update_layout(margin=dict(l=20, r=20, t=30, b=20))
st.plotly_chart(fig)


st.write('## 3D Scatter Plot')
col1, col2, col3, col4 = st.columns(4)
with col1:
    x_axis4 = st.selectbox('Select an x-axis column:', data.columns[:-2])
with col2:
    y_axis4 = st.selectbox('Select a y-axis column:',
                           data.columns[:-2], index=2)
with col3:
    z_axis4 = st.selectbox('Select a y-axis column:',
                           data.columns[:-2], index=3)
with col4:
    color4 = st.selectbox('Select a color column: ', ['species', 'target'])
fig = px.scatter_3d(data, x=x_axis4, y=y_axis4, z=z_axis4, color=color4,
                    title=f'3d scatter plot {x_axis4} vs {y_axis4} vs {z_axis4}', )
fig.update_traces(marker=dict(size=5))
# fig.update_layout(margin=dict(l=20, r=20, t=30, b=20))
st.plotly_chart(fig)


# Show box plot
st.write('## Box Plot')
d_c = data.columns[:-2].copy()
box_column = st.selectbox('Select a column:', d_c)
fig = go.Figure()
fig.add_trace(go.Box(y=data[box_column], name=box_column,
                     boxpoints='all', jitter=0.3, pointpos=-1.8))
fig.update_layout(
    xaxis_title='',
    yaxis_title=box_column,
    xaxis=dict(showgrid=False),
    yaxis=dict(showgrid=False),
    # margin=dict(l=50, r=50, t=30, b=20)
)
st.plotly_chart(fig)

# Create a violin plot of all targets
st.write('## Violin Plot')
fig = px.violin(data.melt(id_vars=['species'], var_name='measurement'),
                y='value', x='measurement', color='species', box=True)
st.plotly_chart(fig)

st.write('## 2D Histogram Contour')
col1, col2, col3 = st.columns(3)
with col1:
    x_axis5 = st.selectbox('Select an x-axis column:   ', data.columns[:-2])
with col2:
    y_axis5 = st.selectbox('Select a y-axis column:   ',
                           data.columns[:-2], index=2)
with col3:
    color5 = st.selectbox('Select a color column:   ', ['species', 'target'])
fig = px.density_contour(data, x=x_axis5, y=y_axis5, color=color5,
                         marginal_x="histogram", marginal_y="histogram",
                         title=f"2D Histogram Contour for {x_axis5} vs {y_axis5}")

st.plotly_chart(fig)

with st.expander("Feature Distribution per Class"):
    column = st.selectbox('Select a column', data.columns[:-2])
    num_bins = st.slider('Number of bins', min_value=5, max_value=50, value=10)

    # Show feature distribution per class
    st.write('## Feature Distribution per Class')
    fig = px.histogram(
        data, x=column, title=f'Distribution of {column} per Class', nbins=num_bins, color='species', barmode='group')
    st.plotly_chart(fig)

st.write('## Scatter Plot matrix')
colors2 = st.selectbox('Select a color column:', ['species', 'target'])
fig = px.scatter_matrix(
    data,
    dimensions=['sepal length (cm)', 'sepal width (cm)',
                'petal length (cm)', 'petal width (cm)'],
    color=colors2,
    title='Scatter plot matrix for Iris dataset',
    opacity=0.7,
)
fig.update_traces(diagonal_visible=False)
fig.update_layout(
    height=800)
st.plotly_chart(fig)

st.write("## Parallel Coordinates Chart")
fig = px.parallel_coordinates(data, color="target",  # color="species",
                              dimensions=['sepal length (cm)', 'sepal width (cm)',
                                          'petal length (cm)', 'petal width (cm)'],
                              color_continuous_scale=px.colors.diverging.RdBu,
                              color_continuous_midpoint=0.5
                              )
st.plotly_chart(fig)

# Show correlation heatmap
st.write('## Correlation Heatmap')
fig = px.imshow(data.corr())
st.plotly_chart(fig)

setosa = data[data['species'] == 'setosa']
versicolor = data[data['species'] == 'versicolor']
virginica = data[data['species'] == 'virginica']
with st.expander("Correlation Heatmaps per class"):

    st.write('### Setosa')
    fig = px.imshow(setosa.corr())
    st.plotly_chart(fig)

    st.write('### Versicolor')
    fig = px.imshow(versicolor.corr())
    st.plotly_chart(fig)

    st.write('### Virginica')
    fig = px.imshow(virginica.corr())
    st.plotly_chart(fig)


# Create a trace for the radar chart
# df = pd.DataFrame({'feature': ['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width'],
#                    'mean_value': [5.8, 3.1, 4.0, 1.3]})
# trace = go.Scatterpolar(
#     r=df['mean_value'].tolist() + [df['mean_value'][0]],
#     theta=df['feature'].tolist() + [df['feature'][0]],
#     fill='toself'
# )

# # Create a layout for the radar chart
# layout = go.Layout(
#     polar=dict(
#         radialaxis=dict(
#             visible=True,
#             range=[0, 6]
#         )
#     ),
#     showlegend=False
# )

# # Create the radar chart
# fig = go.Figure(data=[trace], layout=layout)
# st.plotly_chart(fig)
st.write('## Radar chart (Mean values)')
with st.expander("Radar charts per class"):
    class_means = data.groupby('species').mean()

    st.write('### Setosa')
    mean_values = class_means.loc['setosa']

    # Create a trace for the radar chart
    trace = go.Scatterpolar(
        r=mean_values[:-1].tolist(),
        theta=class_means.columns[:-1],
        fill='toself'
    )

    # Create a layout for the radar chart
    layout = go.Layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 8]
            )
        ),
        showlegend=False
    )

    # Create the radar chart
    fig = go.Figure(data=[trace], layout=layout)
    st.plotly_chart(fig)

    st.write('### Versicolor')
    mean_values = class_means.loc['versicolor']

    # Create a trace for the radar chart
    trace = go.Scatterpolar(
        r=mean_values[:-1].tolist(),
        theta=class_means.columns[:-1],
        fill='toself'
    )
    # Create the radar chart
    fig = go.Figure(data=[trace], layout=layout)
    st.plotly_chart(fig)

    st.write('### Virginica')
    mean_values = class_means.loc['virginica']

    # Create a trace for the radar chart
    trace = go.Scatterpolar(
        r=mean_values[:-1].tolist(),
        theta=class_means.columns[:-1],
        fill='toself'
    )

    # Create the radar chart
    fig = go.Figure(data=[trace], layout=layout)
    st.plotly_chart(fig)

# Footer
st.markdown("---")
st.header("Let's connect!")
link = '[LinkedIn](https://www.linkedin.com/in/tobias-fechner-/)'
st.markdown(link, unsafe_allow_html=True)
link = "[Github](https://github.com/sceeptex)"
st.markdown(link, unsafe_allow_html=True)

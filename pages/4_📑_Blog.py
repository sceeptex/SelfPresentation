from PIL import Image
import streamlit as st
# write your streamlit code here

st.set_page_config(page_title='Tobias Fechner Blog', page_icon="🎓",
                   initial_sidebar_state="collapsed", layout='wide')  # layout='wide'


blog_posts = [
    {
        "title": "Sparks of Artificial General Intelligence: Early experiments with GPT-4",
        "summary": "Microsoft Research has investigated an early version of OpenAI’s GPT-4. They found that this version of GPT-4, along with other large language models (LLMs) such as ChatGPT and Google’s PaLM, exhibit more general intelligence than previous AI models. GPT-4 can solve novel and difficult tasks that span mathematics, coding, vision, medicine, law, psychology and more without needing any special prompting. Its performance is strikingly close to human-level performance and often vastly surpasses prior models such as ChatGPT. The researchers believe that GPT-4 could reasonably be viewed as an early version of an artificial general intelligence (AGI) system. They also discuss the challenges ahead for advancing towards deeper and more comprehensive versions of AGI.",
        "image": "references/posts/agi_puzzle.png",
        "links": {"Paper": "https://arxiv.org/abs/2303.12712"}
    }
    {
        "title": "SimCLR: A Simple Framework for Contrastive Learning of Visual Representations",
        "summary": "Contrastive learning is a technique that aims to learn useful representations of data by comparing similar and dissimilar examples. SimCLR is a framework that simplifies contrastive learning for visual data by using a standard convolutional neural network (CNN) as the encoder and a simple projection head as the contrastive loss function. SimCLR achieves state-of-the-art results on several image classification benchmarks by leveraging large amounts of unlabeled data and data augmentation techniques. SimCLR also demonstrates the benefits of self-supervised pre-training and fine-tuning for downstream tasks.",
        # "image": "references/posts/SimCLR.png",
        "image": "references/posts/SimClrV2.gif",
        "links": {"GitHub": "https://github.com/google-research/simclr", "Paper": "https://arxiv.org/abs/2002.05709"}
    },
    {
        "title": "PyTorch Lightning: Framework for Faster and Easier Deep Learning",
        "summary": "PyTorch Lightning is a deep learning framework that simplifies and accelerates the development of PyTorch models. It provides a high-level interface that abstracts away the boilerplate code and engineering details, allowing you to focus on the research logic and code readability.\n\nOne of the main benefits of PyTorch Lightning is that it enables you to scale your models to run on any hardware (CPU, GPU, or TPU) without changing the source code. You can also use 16-bit precision to train your models faster and with less memory consumption. PyTorch Lightning also automates most of the training loop, such as logging, checkpointing, validation, and testing. Moreover, PyTorch Lightning follows a modular design that decouples the research code from the engineering code, making your models more readable and reproducible.",
        "image": "references/posts/pytorchLighning.png",
        "image2": "references/posts/pytorchAdvantages.png",
        "links": {"Pytorch Lightning": "https://lightning.ai/docs/pytorch/stable/", "Easy Start": "https://lightning.ai/docs/pytorch/stable/model/train_model_basic.html", "YouTube Playlist": "https://www.youtube.com/playlist?list=PLaMu-SDt_RB6-e7GJRQ6cAssjMizOOZUP", "Credit for Graphics": "https://www.assemblyai.com/blog/pytorch-lightning-for-dummies/"}
    },
    {
        "title": "LUX: A Python API for Intelligent Visual Discovery",
        "summary": "Lux is a Python library that facilitate fast and easy data exploration by automating the visualization and data analysis process. By simply printing out a dataframe in a Jupyter notebook, Lux recommends a set of visualizations highlighting interesting trends and patterns in the dataset. Visualizations are displayed via an interactive widget that enables users to quickly browse through large collections of visualizations and make sense of their data.",
        # "image": "references/posts/Lux.png",
        "image": "references/posts/lux.gif",
        "links": {"GitHub": "https://github.com/lux-org/lux", "Notebook Gallery": "https://lux-api.readthedocs.io/en/latest/source/reference/gallery.html"}
    },
    {
        "title": "Tree-Based Pipeline Optimization Tool (TPOT): AutoML for Random Forests",
        "summary": "TPOT stands for Tree-based Pipeline Optimization Tool. It is an open-source library that leverages the popular scikit-learn library for data preprocessing and modeling, and uses a genetic algorithm to search for the best pipeline for a given dataset. A pipeline consists of a series of data transformations and a machine learning model, along with their hyperparameters. TPOT tries to find the optimal combination of these elements by exploring thousands of possible pipelines and evaluating their performance on a cross-validation score.",
        "image": "references/posts/tpot0.png",
        "image2": "references/posts/tpot.png",
        "links": {"GitHub": "https://github.com/EpistasisLab/tpot", "Tutorial and Credit for Graphics": "https://machinelearningmastery.com/tpot-for-automated-machine-learning-in-python/", "Official Website": "http://automl.info/tpot/"}
    },
    {
        "title": "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness",
        "summary": "Attention is a key component of Transformer models, but it is also slow and memory-hungry on long sequences. FlashAttention is a new algorithm that reorders the attention computation and leverages classical techniques (tiling, recomputation) to significantly speed it up and reduce memory usage from quadratic to linear in sequence length. FlashAttention also accounts for the IO complexity of attention, which is often overlooked by existing methods. FlashAttention can train Transformers faster than existing baselines on various tasks and sequence lengths, and enable longer context in Transformers, yielding higher quality models and entirely new capabilities.",
        "image": "references/posts/flashattn_banner.jpg",
        "links": {"GitHub": "https://github.com/HazyResearch/flash-attention", "Paper": "https://arxiv.org/abs/2205.14135"}
    },

    # add more blog posts here
]
columns = st.columns([1.5, 5, 1, 5, 1.5])
with columns[1]:
    st.title("My Data Science Blog")

st.write("")

columns = st.columns([1.5, 5, 1, 5, 1.5])

# col1, col2, col3, col4, col5 = st.columns([2, 5, 1, 5, 2])

for i, blog_post in enumerate(blog_posts):
    # select the column based on the index
    column = columns[(i % 2)*2+1]

    column.markdown("## {}".format(blog_post['title']))
    column.image(blog_post["image"], use_column_width=True)
    if "image2" in blog_post.keys():
        column.image(blog_post["image2"], use_column_width=True)
    column.markdown(blog_post["summary"])

    # show links
    markdown_text = ""
    for i, (key, value) in enumerate(blog_post["links"].items()):
        if i+1 < len(blog_post["links"]) and len(blog_post["links"]) != 1:

            markdown_text += '[{}]({}), '.format(key, value)
        else:
            markdown_text += '[{}]({})'.format(key, value)
    column.markdown(markdown_text, unsafe_allow_html=True)


st.write("")
st.write("")
st.write("")
st.write("")
st.markdown('---')


def clickable_card(image_url: str, redirect_url: str):
    # Load image and create card
    image = Image.open(image_url)
    st.image(image, width=300, use_column_width=True)
    card = st.empty()
    card.markdown(f'<div style="border-radius: 10px; padding: 10px;">'
                  f'<a href="{redirect_url}" target="_blank">'
                  f'<h3 style="margin: 0;">Click Here</h3></a></div>', unsafe_allow_html=True)


columns = st.columns([1.5, 5, 1, 5, 1.5])
with columns[1]:
    st.header("Let's connect!")
columns = st.columns([1.5, 5, 1, 5, 1.5])
with columns[1]:
    clickable_card("references/linkedin.png",
                   'https://www.linkedin.com/in/tobias-fechner-/')
with columns[3]:
    clickable_card("references/GitHub2.jpeg", 'https://github.com/sceeptex')

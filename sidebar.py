import streamlit as st
def render_sidebar():
    st.sidebar.image("./static/logo.png", width=200)
    st.sidebar.title("Navigation")
    pages = [
        "Home",
        "Detection",
        "Contact Us",
        "About Project",
        "Project Summary",
        "Analysis",
        "Result Analysis"
    ]
    selected_page = st.sidebar.radio("Go to", pages)
    return selected_page

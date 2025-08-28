import streamlit as st
import authlib 
from app import run_app

IMAGE_ADDRESS="https://img.goodfon.com/original/1920x1080/8/71/fireman-fire-man.jpg"


if not st.user.is_logged_in:
    st.title("Login with Google")
    st.image(IMAGE_ADDRESS)
    if st.sidebar.button("Log in with Google", type="primary", icon=":material/login:"):
        st.login() 

else:
    if st.sidebar.button("Log Out with Google", type="secondary", icon=":material/logout:"):
        st.logout()
    
    run_app()

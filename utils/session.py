import streamlit as st

def initialize_session_state():
    """
    Initialize session state variables if they don't exist.
    """
    if 'current_page' not in st.session_state:
        st.session_state.current_page = "Home"
    
    if 'data' not in st.session_state:
        st.session_state.data = None
    
    if 'channels' not in st.session_state:
        st.session_state.channels = []
    
    if 'outcome_column' not in st.session_state:
        st.session_state.outcome_column = None
    
    if 'date_column' not in st.session_state:
        st.session_state.date_column = None
    
    if 'adstock_params' not in st.session_state:
        st.session_state.adstock_params = {}
    
    if 'saturation_params' not in st.session_state:
        st.session_state.saturation_params = {}
    
    if 'model' not in st.session_state:
        st.session_state.model = None
    
    if 'attribution' not in st.session_state:
        st.session_state.attribution = None

def get_nav_page():
    """
    Get the current navigation page from session state.
    """
    return st.session_state.current_page

def set_nav_page(page):
    """
    Set the current navigation page in session state.
    
    Parameters:
    -----------
    page : str
        Page name to navigate to
    """
    st.session_state.current_page = page

def clear_session_state():
    """
    Clear all session state variables except the current page.
    """
    current_page = st.session_state.current_page
    
    for key in list(st.session_state.keys()):
        if key != 'current_page':
            del st.session_state[key]
    
    # Reinitialize the session state
    initialize_session_state()
    
    # Restore the current page
    st.session_state.current_page = current_page
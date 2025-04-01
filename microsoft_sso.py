import streamlit as st
import requests
import msal
import json
import os
from datetime import datetime

# Microsoft SSO Configuration
def setup_microsoft_sso():
    """Configure Microsoft SSO integration"""
    # These values should be moved to environment variables or secrets in production
    MS_CLIENT_ID = os.environ.get("MS_CLIENT_ID", "50053a62-e48a-4745-91ce-d2af1c30d445")
    MS_CLIENT_SECRET = os.environ.get("MS_CLIENT_SECRET", "d52061d6-d3f2-40e4-a1df-fed21d8e79f2")
    MS_TENANT_ID = os.environ.get("MS_TENANT_ID", "d52061d6-d3f2-40e4-a1df-fed21d8e79f2")
    MS_REDIRECT_URI = os.environ.get("MS_REDIRECT_URI", "https://bogo-autoeval.streamlit.app/")
    MS_AUTHORITY = f"https://login.microsoftonline.com/{MS_TENANT_ID}"
    
    # Define the scopes your app needs
    SCOPES = ["User.Read", "email", "profile", "openid"]
    
    return {
        "client_id": MS_CLIENT_ID,
        "client_secret": MS_CLIENT_SECRET,
        "authority": MS_AUTHORITY,
        "redirect_uri": MS_REDIRECT_URI,
        "scopes": SCOPES
    }

def initialize_msal_app(config):
    """Initialize the MSAL application for authentication"""
    return msal.ConfidentialClientApplication(
        config["client_id"],
        authority=config["authority"],
        client_credential=config["client_secret"]
    )

def get_auth_url(msal_app, config):
    """Generate the authorization URL for sign-in"""
    return msal_app.get_authorization_request_url(
        config["scopes"],
        redirect_uri=config["redirect_uri"],
        state=json.dumps({"streamlit": "auth"}),
        prompt="select_account"
    )

def get_token_from_code(msal_app, code, config):
    """Acquire token using the authorization code"""
    return msal_app.acquire_token_by_authorization_code(
        code,
        scopes=config["scopes"],
        redirect_uri=config["redirect_uri"]
    )

def get_user_info(access_token):
    """Get user information from Microsoft Graph API"""
    headers = {"Authorization": f"Bearer {access_token}"}
    graph_data = requests.get(
        "https://graph.microsoft.com/v1.0/me",
        headers=headers
    ).json()
    
    return graph_data

def logout():
    """Clear session and logout user"""
    for key in list(st.session_state.keys()):
        if key in ["user_info", "token_cache", "access_token", "is_authenticated"]:
            del st.session_state[key]

def check_user_belongs_to_org(user_info, allowed_domains=None):
    """Check if the user belongs to allowed organization domains"""
    if not allowed_domains:
        # Default: allow any Microsoft account
        return True
    
    user_email = user_info.get("mail") or user_info.get("userPrincipalName", "")
    if not user_email:
        return False
    
    domain = user_email.split('@')[-1].lower()
    return domain in [d.lower() for d in allowed_domains]

def auth_required(func):
    """Decorator to ensure user is authenticated before accessing protected pages"""
    def wrapper(*args, **kwargs):
        if not st.session_state.get("is_authenticated"):
            st.error("Please login to access this page")
            display_login()
            return None
        return func(*args, **kwargs)
    return wrapper

def display_login():
    """Display the login button and handle the authentication flow"""
    # Initialize SSO configuration
    sso_config = setup_microsoft_sso()
    msal_app = initialize_msal_app(sso_config)
    
    # Check if we're handling a callback with code
    query_params = st.experimental_get_query_params()
    if "code" in query_params:
        code = query_params["code"][0]
        # Clear the URL to avoid refreshing issues
        st.experimental_set_query_params()
        
        try:
            # Get token using the authorization code
            result = get_token_from_code(msal_app, code, sso_config)
            if "access_token" in result:
                st.session_state.access_token = result["access_token"]
                st.session_state.is_authenticated = True
                
                # Get user info
                user_info = get_user_info(result["access_token"])
                st.session_state.user_info = user_info
                
                # Check if user belongs to allowed organization
                allowed_domains = ["yourcompany.com"]  # Configure your allowed domains
                if not check_user_belongs_to_org(user_info, allowed_domains):
                    st.error("You do not have permission to access this application. Please contact your administrator.")
                    logout()
                    st.stop()
                
                st.success(f"Welcome, {user_info.get('displayName')}!")
                st.rerun()  # Rerun to refresh the page
            else:
                st.error(f"Authentication failed: {result.get('error_description', 'Unknown error')}")
        except Exception as e:
            st.error(f"Error during authentication: {str(e)}")
    
    # If user is not authenticated, show the login button
    if not st.session_state.get("is_authenticated"):
        auth_url = get_auth_url(msal_app, sso_config)
        st.markdown(f"""
        <div style='text-align: center; margin-top: 50px;'>
            <h1>Welcome to the Auto-Eval Selection & Evaluation Demo</h1>
            <p>Please login with your Microsoft account to continue</p>
            <a href='{auth_url}' target='_self'>
                <button style='
                    background-color: #0078d4;
                    color: white;
                    padding: 10px 20px;
                    border: none;
                    border-radius: 4px;
                    cursor: pointer;
                    font-size: 16px;
                '>
                    Login with Microsoft
                </button>
            </a>
        </div>
        """, unsafe_allow_html=True)
        st.stop()  # Stop execution here

def display_user_info():
    """Display the user information and logout button in the sidebar"""
    if st.session_state.get("is_authenticated") and st.session_state.get("user_info"):
        user_info = st.session_state.user_info
        with st.sidebar:
            st.write(f"Logged in as: **{user_info.get('displayName')}**")
            st.write(f"Email: {user_info.get('mail') or user_info.get('userPrincipalName', 'N/A')}")
            
            if st.button("Logout"):
                logout()
                st.rerun()

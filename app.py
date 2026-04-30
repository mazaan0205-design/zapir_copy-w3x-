import streamlit as st
import os
import datetime
from sqlalchemy import create_engine, Column, String, DateTime, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from streamlit_google_auth import Authenticate

# --- 1. SQL DATABASE SETUP ---
# Using SQLite because it's a single file (vortex_agent.db)
engine = create_engine("sqlite:///vortex_agent.db")
Base = declarative_base()

class User(Base):
    __tablename__ = 'users'
    email = Column(String, primary_key=True)
    name = Column(String)
    last_login = Column(DateTime, default=datetime.datetime.utcnow)
    access_token = Column(Text)

Base.metadata.create_all(engine)
Session = sessionmaker(bind=engine)
db_session = Session()

# --- 2. AUTHENTICATION SETUP ---
auth = Authenticate(
    secret_credentials_path='google_credentials.json', # You need this file
    cookie_name='vortex_auth_cookie',
    cookie_key='vortex_secret_key',
    redirect_uri=st.secrets["google_auth"]["redirect_uri"],
)

# --- 3. HELPER: SAVE TO SQL ---
def sync_user_to_sql(user_info, token):
    email = user_info.get("email")
    user = db_session.query(User).filter_by(email=email).first()
    
    if not user:
        user = User(
            email=email, 
            name=user_info.get("name"), 
            access_token=token
        )
        db_session.add(user)
    else:
        user.access_token = token
        user.last_login = datetime.datetime.utcnow()
    
    db_session.commit()

# --- 4. THE APP INTERFACE ---
auth.check_authenticity()

if not st.session_state.get("auth_status"):
    st.title("Vortex.ai Agent")
    st.info("Please login to access your AI Automation Agent.")
    auth.login()
else:
    # Get user info from Google
    user_data = st.session_state["user_info"]
    token = st.session_state.get("token") # Depends on your auth library version
    
    # Save/Update in SQL
    sync_user_to_sql(user_data, token)
    
    st.sidebar.success(f"Connected: {user_data['email']}")
    
    # --- 5. AGENT CHAT INTERFACE ---
    st.title("Vortex.ai Operations")
    
    # Your LangChain Agent logic goes here, 
    # using user_data['email'] for all actions!

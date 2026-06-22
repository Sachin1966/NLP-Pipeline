import streamlit as st
import datetime
from src.database.connection import SessionLocal
from src.database.models import User
from src.api.auth import get_password_hash, verify_password

def render_auth_page():
    """Renders registration and login tabs using Streamlit."""
    st.markdown("""
    <div style="text-align: center; margin-bottom: 20px;">
        <h1 style="color: #1E3A8A; font-size: 2.5rem; margin-bottom: 5px;">⚡ Voice Intelligence Engine</h1>
        <p style="color: #4B5563; font-size: 1.1rem;">Production-Grade Customer Feedback & MLOps Platform</p>
    </div>
    """, unsafe_allow_html=True)
    
    auth_tab1, auth_tab2 = st.tabs(["🔒 Sign In", "📝 Create Account"])
    
    with auth_tab1:
        st.subheader("Login Credentials")
        login_user = st.text_input("Username", key="login_usr")
        login_pass = st.text_input("Password", type="password", key="login_pwd")
        
        if st.button("Sign In", type="primary", use_container_width=True):
            if not login_user or not login_pass:
                st.error("Please fill in both username and password fields.")
            else:
                db = SessionLocal()
                try:
                    user = db.query(User).filter(User.username == login_user).first()
                    if user and verify_password(login_pass, user.password_hash):
                        st.session_state["authenticated"] = True
                        st.session_state["username"] = user.username
                        st.session_state["role"] = user.role
                        st.success("Successfully logged in!")
                        st.rerun()
                    else:
                        st.error("Invalid username or password.")
                finally:
                    db.close()
                    
    with auth_tab2:
        st.subheader("Registration Profile")
        reg_user = st.text_input("Choose Username", key="reg_usr")
        reg_pass = st.text_input("Choose Password", type="password", key="reg_pwd")
        reg_role = st.selectbox("Assign Role", ["user", "manager", "admin"], key="reg_role")
        
        if st.button("Sign Up", use_container_width=True):
            if not reg_user or not reg_pass:
                st.error("Please specify both username and password.")
            else:
                db = SessionLocal()
                try:
                    existing = db.query(User).filter(User.username == reg_user).first()
                    if existing:
                        st.error("Username is already taken.")
                    else:
                        hashed_pw = get_password_hash(reg_pass)
                        new_user = User(
                            username=reg_user,
                            password_hash=hashed_pw,
                            role=reg_role,
                            created_at=datetime.datetime.utcnow()
                        )
                        db.add(new_user)
                        db.commit()
                        st.success("Account created successfully! Please sign in using the login tab.")
                finally:
                    db.close()

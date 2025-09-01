import streamlit as st
import json
import os
import subprocess
import sys
import re

# File to store user credentials
USER_FILE = "users.json"

# Load existing users
if os.path.exists(USER_FILE):
    with open(USER_FILE, "r") as f:
        users = json.load(f)
else:
    users = {}

# Save users back to file
def save_users():
    with open(USER_FILE, "w") as f:
        json.dump(users, f, indent=4)

# Password validation
def valid_password(pw: str) -> bool:
    if len(pw) < 8:
        st.error("⚠️ Password must be at least 8 characters long.")
        return False
    if not re.search(r"[0-9]", pw):
        st.error("⚠️ Password must contain at least one number.")
        return False
    if not re.search(r"[A-Z]", pw):
        st.error("⚠️ Password must contain at least one uppercase letter.")
        return False
    if not re.search(r"[a-z]", pw):
        st.error("⚠️ Password must contain at least one lowercase letter.")
        return False
    return True


def login_page():
    st.set_page_config(page_title="CareSync Login", page_icon="", layout="centered")
    st.markdown("<h2 style='text-align:center;'>🩺 CareSync</h2>", unsafe_allow_html=True)

    # Track which form to show
    if "show_signup" not in st.session_state:
        st.session_state.show_signup = False

    if not st.session_state.show_signup:
        # ---------------- LOGIN FORM ----------------
        st.subheader("Sign in")
        username = st.text_input("Username", placeholder="Enter your username")
        password = st.text_input("Password", type="password", placeholder="Enter your password")

        if st.button("Login", use_container_width=True):
            if username in users and users[username]["password"] == password:
                st.session_state["logged_in"] = True
                st.session_state["username"] = username
                st.session_state["full_name"] = users[username]["name"]
                st.success(f"Welcome back, {users[username]['name']}! Redirecting...")
                st.session_state["run_chatbot"] = True
                st.rerun()
            else:
                st.error("❌ Invalid username or password.")

        st.markdown("---")
        if st.button("Create an account", type="secondary", use_container_width=True):
            st.session_state.show_signup = True
            st.rerun()

    else:
        # ---------------- SIGNUP FORM ----------------
        st.subheader("Create your CareSync account")
        full_name = st.text_input("Full Name", placeholder="John Doe")
        username = st.text_input("Choose a Username", placeholder="johndoe123")
        password = st.text_input(
            "Password",
            type="password",
            placeholder="Enter a strong password",
            help="At least 8 characters, with uppercase, lowercase, and a number"
        )
        confirm_pw = st.text_input("Confirm Password", type="password", placeholder="Re-enter password")

        if st.button("Sign Up", use_container_width=True):
            if username in users:
                st.error("❌ Username already exists.")
            elif password != confirm_pw:
                st.error("❌ Passwords do not match.")
            elif not valid_password(password):
                pass
            else:
                users[username] = {"name": full_name, "password": password}
                save_users()
                st.success("✅ Account created! Please login.")
                st.session_state.show_signup = False
                st.rerun()

        st.markdown("---")
        if st.button("Already have an account? Sign in", type="secondary", use_container_width=True):
            st.session_state.show_signup = False
            st.rerun()


# ----------------- MAIN APP -----------------
if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False
if "run_chatbot" not in st.session_state:
    st.session_state["run_chatbot"] = False

if not st.session_state["logged_in"]:
    login_page()
else:
    caresync_page = os.path.join("pages", "caresync.py")
    if os.path.exists(caresync_page):
        try:
            st.switch_page("pages/caresync.py")
        except Exception as e:
            st.error(f"⚠️ Could not switch page: {e}")
    else:
        st.info("Launching CareSync chatbot in a new tab...")
        subprocess.Popen([sys.executable, "caresync.py"])
        st.stop()

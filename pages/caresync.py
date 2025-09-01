import os
import time
import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_community.chat_models import ChatOpenAI
from dotenv import load_dotenv
load_dotenv()
import requests
from geopy.geocoders import Nominatim
from geopy.distance import geodesic
import geocoder
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium
import math

# ---- Overpass helpers ----
OVERPASS_ENDPOINTS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://lz4.overpass-api.de/api/interpreter",
]

def build_overpass_query(lat: float, lon: float, radius_m: int, types: list[str]) -> str:
    # OSM tags for types
    type_to_tag = {
        "hospital": '["amenity"="hospital"]',
        "clinic": '["amenity"="clinic"]',
        "pharmacy": '["amenity"="pharmacy"]',
        "doctors": '["amenity"="doctors"]',
    }
    parts = []
    for t in types:
        tag = type_to_tag[t]
        parts.append(f'node{tag}(around:{radius_m},{lat},{lon});')
        parts.append(f'way{tag}(around:{radius_m},{lat},{lon});')
    body = "\n".join(parts)
    return f"""[out:json][timeout:25];
(
{body}
);
out center;"""

@st.cache_data(show_spinner=False, ttl=300)
def overpass_search(lat: float, lon: float, radius_m: int, types: tuple[str, ...]):
    query = build_overpass_query(lat, lon, radius_m, list(types))
    last_err = None
    for ep in OVERPASS_ENDPOINTS:
        try:
            r = requests.post(ep, data={"data": query}, timeout=30)
            if r.status_code == 200:
                return r.json()
            last_err = f"{r.status_code} {r.text[:200]}"
        except Exception as e:
            last_err = str(e)
    raise RuntimeError(f"Overpass request failed: {last_err}")

def element_latlon(el):
    if "lat" in el and "lon" in el:
        return el["lat"], el["lon"]
    if "center" in el and "lat" in el["center"] and "lon" in el["center"]:
        return el["center"]["lat"], el["center"]["lon"]
    return None, None

def pretty_address(tags: dict) -> str:
    parts = [tags.get("addr:street"), tags.get("addr:city"), tags.get("addr:state")]
    return ", ".join([p for p in parts if p])



DB_FAISS_PATH = "vectorstore/db_faiss"

@st.cache_resource
def get_vectorstore():
    embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    return db

def set_custom_prompt(custom_prompt_template):
    prompt = PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])
    return prompt

def load_llm():
    return ChatOpenAI(
        openai_api_base="https://openrouter.ai/api/v1",
        openai_api_key=os.getenv("OPENAI_API_KEY"),  
        model="mistralai/mistral-7b-instruct",
        temperature=0.5
    )


def main():
    st.set_page_config(page_title="CareSync Chatbot", page_icon="🩺", layout="wide")

    # ------------------- LOGIN PROTECTION -------------------
    if "logged_in" not in st.session_state or not st.session_state["logged_in"]:
        st.warning("⚠️ Please login to access CareSync.")
        st.stop()

    # ------------------- SIDEBAR -------------------
    with st.sidebar:
        st.markdown("Navigation")

        menu = st.radio(
            "Choose an option:",
            ["Chatbot", "Nearby Facilities"],
            index=0,
            label_visibility="collapsed"
        )

        st.markdown("---")

        logout_placeholder = st.empty()
        with logout_placeholder.container():
            if st.button("Logout", key="logout", use_container_width=True):
                st.session_state["logged_in"] = False
                st.session_state["username"] = None
                st.session_state["full_name"] = None
                st.session_state["run_chatbot"] = False
                st.success("You have been logged out.")
                st.rerun()

    # ------------------- HEADER -------------------
    if "full_name" in st.session_state and st.session_state["full_name"]:
        st.markdown(
            f"<h2 style='text-align:center;'>🩺 CareSync — Hello, {st.session_state['full_name']}!</h2>",
            unsafe_allow_html=True
        )
    else:
        st.markdown("<h2 style='text-align:center;'>🩺 CareSync </h2>", unsafe_allow_html=True)

    # ------------------- MAIN CONTENT -------------------
    if menu == "Chatbot":
        st.title("Ask CareSync AI")

        # Initialize chat messages for UI display
        if 'messages' not in st.session_state:
            st.session_state.messages = []

        # Initialize memory for LLM (conversation history)
        if "memory" not in st.session_state:
            st.session_state.memory = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True
            )

        # Display previous messages
        for message in st.session_state.messages:
            st.chat_message(message['role']).markdown(message['content'])

        #  Always show chat input
        prompt = st.chat_input("💬 Ask me anything about healthcare...")

        #  Clear chat button in sidebar
        with st.sidebar:
            if st.button(" Clear Chat"):
                st.session_state.messages = []
                st.session_state.memory.clear()
                st.rerun()

        if prompt:
            st.chat_message('user').markdown(prompt)
            st.session_state.messages.append({'role': 'user', 'content': prompt})

            CUSTOM_PROMPT_TEMPLATE = """
            Use the pieces of information provided in the context to answer user's question.
            If you don't know the answer, just say that you don't know. Don't make up an answer. 
            Don't provide anything outside of the given context.

            Context: {context}
            Question: {question}

            Start the answer directly. No small talk please.
            """

            try:
                vectortore = get_vectorstore()
                if vectortore is None:
                    st.error("Failed to Load vectorstore")

                qa_chain = ConversationalRetrievalChain.from_llm(
                    llm=load_llm(),
                    retriever=vectortore.as_retriever(search_kwargs={'k': 3}),
                    memory=st.session_state.memory,
                    combine_docs_chain_kwargs={'prompt': set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
                )

                response = qa_chain({"question": prompt})
                result = response["answer"]

                message_placeholder = st.chat_message("assistant").empty()
                full_response = ""
                for chunk in result.split():
                    full_response += chunk + " "
                    message_placeholder.markdown(full_response + "▌")  # cursor effect
                    time.sleep(0.05)  # adjust speed
                # Final response (remove cursor)
                message_placeholder.markdown(full_response)
                st.session_state.messages.append({'role': 'assistant', 'content': full_response})
            except Exception as e:
                st.error(f"Error: {str(e)} Something Went Wrong!")

    elif menu == "Nearby Facilities":
        st.title(" Nearby Healthcare Facilities")

        # --- state for location ---
        if "location_latlon" not in st.session_state:
            st.session_state.location_latlon = None
            st.session_state.location_label = None

        # Controls
        col1, col2 = st.columns([2, 1])
        with col1:
            use_ip = st.button("Use My Current Location (IP)")
        with col2:
            radius_km = st.slider("Search radius (km)", 1, 10, 3)

        # Manual location input
        location_input = st.text_input("Or enter your location (e.g., 'Kolkata, India'):")

        # Detect via IP
        if use_ip:
            g = geocoder.ip("me")
            if g.ok and g.latlng:
                st.session_state.location_latlon = (g.latlng[0], g.latlng[1])
                st.session_state.location_label = f"{g.city}, {g.country}" if g.city else "Current location (IP)"
                st.success(f"Detected: {st.session_state.location_label}")
            else:
                st.error("❌ Could not detect your location automatically.")

        # Geocode manual input
        if location_input:
            geolocator = Nominatim(user_agent="caresync_app")  # set your own UA
            loc = geolocator.geocode(location_input)
            if loc:
                st.session_state.location_latlon = (loc.latitude, loc.longitude)
                st.session_state.location_label = loc.address
                st.success(f"Location found: {loc.address}")
            else:
                st.error("❌ Location not found.")

        # Filters
        st.markdown("**Filter facility types:**")
        c1, c2, c3, c4 = st.columns(4)
        with c1: f_h = st.checkbox("Hospitals", True)
        with c2: f_c = st.checkbox("Clinics", True)
        with c3: f_p = st.checkbox("Pharmacies", True)
        with c4: f_d = st.checkbox("Doctors", True)

        selected_types = []
        if f_h: selected_types.append("hospital")
        if f_c: selected_types.append("clinic")
        if f_p: selected_types.append("pharmacy")
        if f_d: selected_types.append("doctors")

        if not selected_types:
            st.info("Select at least one facility type.")
            st.stop()

        # If we have a location, search OSM
        if st.session_state.location_latlon:
            lat, lon = st.session_state.location_latlon
            radius_m = int(radius_km * 1000)

            with st.spinner("Searching nearby facilities…"):
                try:
                    data = overpass_search(lat, lon, radius_m, tuple(selected_types))
                except Exception as e:
                    st.error(f"Overpass error: {e}")
                    st.stop()

            elements = data.get("elements", [])
            results = []

            for el in elements:
                tags = el.get("tags", {})
                name = tags.get("name", "Unnamed facility")
                ttype = tags.get("amenity", "unknown")
                fac_lat, fac_lon = element_latlon(el)
                if fac_lat is None:  # skip if no coordinates
                    continue

                dist_km = geodesic((lat, lon), (fac_lat, fac_lon)).km
                addr = pretty_address(tags)
                phone = tags.get("phone") or tags.get("contact:phone")
                website = tags.get("website") or tags.get("contact:website")

                results.append({
                    "name": name,
                    "type": ttype,
                    "lat": fac_lat,
                    "lon": fac_lon,
                    "distance_km": dist_km,
                    "address": addr,
                    "phone": phone,
                    "website": website
                })

            # Sort by distance, keep top N (to avoid clutter)
            results.sort(key=lambda x: x["distance_km"])
            max_show = 50
            results = results[:max_show]

            # Map
            m = folium.Map(location=[lat, lon], zoom_start=14)
            # user marker
            folium.Marker([lat, lon], tooltip=st.session_state.location_label or "You are here",
                        icon=folium.Icon(color="blue", icon="info-sign")).add_to(m)
            # cluster
            cluster = MarkerCluster().add_to(m)

            color_by_type = {"hospital": "red", "clinic": "orange", "pharmacy": "green", "doctors": "cadetblue"}

            for r in results:
                ic_color = color_by_type.get(r["type"], "gray")
                popup_lines = [f"<b>{r['name']}</b>", f"{r['type'].title()}"]
                if r["address"]: popup_lines.append(r["address"])
                popup_lines.append(f"{r['distance_km']:.2f} km away")
                if r["phone"]: popup_lines.append(f"☎ {r['phone']}")
                if r["website"]: popup_lines.append(f"<a href='{r['website']}' target='_blank'>Website</a>")
                folium.Marker(
                    [r["lat"], r["lon"]],
                    tooltip=r["name"],
                    popup=folium.Popup("<br>".join(popup_lines), max_width=300),
                    icon=folium.Icon(color=ic_color, icon="plus-sign")
                ).add_to(cluster)

            st_folium(m, width=800, height=520)

            # List view/table
            if results:
                st.subheader("📋 Nearby Facilities")
                for r in results:
                    line = f"**{r['name']}** — {r['type'].title()} • {r['distance_km']:.2f} km"
                    if r["address"]:
                        line += f"\n\n{r['address']}"
                    if r["phone"]:
                        line += f"\n\n☎ {r['phone']}"
                    if r["website"]:
                        line += f"\n\n🔗 {r['website']}"
                    st.markdown(line)
                    st.markdown("---")
            else:
                st.warning("No facilities found within the selected radius. Try increasing it.")
        else:
            st.info("Use ** Use My Current Location (IP)** or type a location to begin.")


# ------------------- ENTRY POINT -------------------
if __name__ == "__main__":
    main()

import os
import sys
import glob
import importlib
import subprocess

import streamlit as st

from bark_infinity import api, config

logger = config.logger

REQUIREMENTS = "requirements-pip.txt"


def auto_install():
    """Install missing packages listed in REQUIREMENTS."""
    if not os.path.exists(REQUIREMENTS):
        return
    packages = []
    with open(REQUIREMENTS, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                packages.append(line)
    # ensure streamlit is included
    if "streamlit" not in packages:
        packages.append("streamlit")

    missing = []
    for pkg in packages:
        mod_name = pkg.split("==")[0].split(">")[0].replace("-", "_")
        try:
            importlib.import_module(mod_name)
        except ImportError:
            missing.append(pkg)
    if missing:
        st.sidebar.write("Installing required packages ...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", *missing])
        except Exception as e:
            st.sidebar.error(f"Auto installation failed: {e}")
        st.sidebar.write("Please restart the app after installation")
        st.stop()


def get_npz_files():
    directories = ["custom_speakers/", "bark/assets/prompts/"]
    files = []
    for d in directories:
        files.extend(glob.glob(os.path.join(d, "**", "*.npz"), recursive=True))
    return sorted(files)


def main():
    st.set_page_config(page_title="Bark Infinity", page_icon="🎶", layout="wide")
    st.markdown(
        """
        <style>
            body, .stApp { background-color: #0e1117; color: white; }
        </style>
        """,
        unsafe_allow_html=True,
    )
    auto_install()

    st.title("Bark Infinity Streamlit")

    with st.sidebar:
        st.header("Configuration")
        log_level = st.selectbox("Log Level", ["INFO", "DEBUG", "WARNING", "ERROR"], index=0)
        logger.setLevel(log_level)
        speaker_files = get_npz_files()
        speaker = st.selectbox("Speaker", [""] + speaker_files)
        text_temp = st.slider("Text Temperature", 0.0, 1.0, 0.7)
        waveform_temp = st.slider("Waveform Temperature", 0.0, 1.0, 0.7)

    prompt = st.text_area("Prompt", help="Main prompt text")
    lyrics = st.text_area("Lyrics (optional)")
    genre = st.text_input("Genre/Style", "")

    if st.button("Generate"):
        full_text = prompt
        if lyrics:
            full_text += "\n" + lyrics
        if genre:
            full_text += f"\nIn the style of {genre}."
        kwargs = {
            "text_prompt": full_text,
            "history_prompt": speaker if speaker else None,
            "text_temp": text_temp,
            "waveform_temp": waveform_temp,
        }
        st.write("Generating audio ...")
        filename = api.generate_audio_long_from_gradio(**kwargs)
        if filename and os.path.exists(filename):
            st.audio(filename)
            st.success(f"Saved to {filename}")
        else:
            st.error("Failed to generate audio")


if __name__ == "__main__":
    main()

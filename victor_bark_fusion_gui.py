import os
import sys
import glob
import importlib
import threading
import time
import queue

import streamlit as st

from bark_infinity import api, config, error_handling

# VictorPrimeEmergentFusionMonolithGUI-STABLE.py contains dashes in the
# filename, so we have to load it dynamically.
spec = importlib.util.spec_from_file_location(
    "victor_prime", os.path.join(os.path.dirname(__file__), "VictorPrimeEmergentFusionMonolithGUI-STABLE.py")
)
victor_prime = importlib.util.module_from_spec(spec)
spec.loader.exec_module(victor_prime)

NodeRegistry = victor_prime.NodeRegistry
HiveMind = victor_prime.HiveMind
CognitiveCore = victor_prime.CognitiveCore
LyricalEngine = victor_prime.LyricalEngine
MicrophoneInput = victor_prime.MicrophoneInput
AudioSynthesis = victor_prime.AudioSynthesis

logger = config.logger
error_handling.set_global_exception_logger()

REQUIREMENTS = "requirements-pip.txt"


def auto_install():
    if not os.path.exists(REQUIREMENTS):
        return
    packages = []
    with open(REQUIREMENTS, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                packages.append(line)
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
            import subprocess
            subprocess.check_call([sys.executable, "-m", "pip", "install", *missing])
            st.sidebar.success("Packages installed. Please restart the app.")
        except Exception as e:
            st.sidebar.error(f"Auto installation failed: {e}")
        st.stop()


def get_npz_files():
    directories = ["custom_speakers/", "bark/assets/prompts/"]
    files = []
    for d in directories:
        files.extend(glob.glob(os.path.join(d, "**", "*.npz"), recursive=True))
    return sorted(files)


class HiveController:
    def __init__(self):
        self.log_queue = queue.Queue()
        self.hive = HiveMind(self.log_queue)
        NodeRegistry.register_node(CognitiveCore.NODE_NAME, CognitiveCore)
        NodeRegistry.register_node(LyricalEngine.NODE_NAME, LyricalEngine)
        NodeRegistry.register_node(MicrophoneInput.NODE_NAME, MicrophoneInput)
        NodeRegistry.register_node(AudioSynthesis.NODE_NAME, AudioSynthesis)
        self.hive.evolve()
        self.running_event = threading.Event()
        self.logs = []

    def _run(self, steps, delay):
        for _ in range(steps):
            if not self.running_event.is_set():
                break
            self.hive.evolve()
            time.sleep(delay)
        self.running_event.clear()

    def start(self, steps, delay):
        if not self.running_event.is_set():
            self.running_event.set()
            threading.Thread(target=self._run, args=(steps, delay)).start()

    def stop(self):
        self.running_event.clear()

    def step_once(self):
        if not self.running_event.is_set():
            self.hive.evolve()

    def get_logs(self):
        while not self.log_queue.empty():
            self.logs.append(self.log_queue.get())
        self.logs = self.logs[-200:]
        return list(self.logs)


hive_controller = HiveController()


def main():
    st.set_page_config(page_title="Victor Bark Fusion", page_icon="⚡", layout="wide")
    st.markdown(
        """
        <style>
            body, .stApp { background-color: #0e1117; color: white; }
        </style>
        """,
        unsafe_allow_html=True,
    )
    auto_install()

    col1, col2 = st.columns(2)

    with col1:
        st.header("Bark Infinity Audio")
        with st.form(key="bark_form"):
            speaker_files = get_npz_files()
            speaker = st.selectbox("Speaker", ["None"] + speaker_files)
            text_temp = st.slider("Text Temperature", 0.0, 1.0, 0.7)
            waveform_temp = st.slider("Waveform Temperature", 0.0, 1.0, 0.7)
            prompt = st.text_area("Prompt")
            submitted = st.form_submit_button("Generate")
        if submitted:
            kwargs = {
                "text_prompt": prompt,
                "history_prompt": None if speaker == "None" else speaker,
                "text_temp": text_temp,
                "waveform_temp": waveform_temp,
            }
            st.write("Generating audio ...")
            safe_generate = error_handling.retry_on_exception()(api.generate_audio_long_from_gradio)
            filename = safe_generate(**kwargs)
            if filename and os.path.exists(filename):
                st.audio(filename)
                st.success(f"Saved to {filename}")
            else:
                st.error("Failed to generate audio")

    with col2:
        st.header("Victor Prime Hive")
        steps = st.slider("Evolution Steps", 1, 50, 5)
        delay = st.slider("Step Delay (s)", 0.1, 2.0, 0.2, 0.1)
        c1, c2, c3 = st.columns(3)
        if c1.button("Run"):
            hive_controller.start(steps, delay)
        if c2.button("Stop"):
            hive_controller.stop()
        if c3.button("Step"):
            hive_controller.step_once()
        st.metric("Current Generation", hive_controller.hive.generation)
        st.metric("Active Agents", len(hive_controller.hive.agents))
        log_lines = hive_controller.get_logs()
        st.text_area("Event Log", "\n".join(reversed(log_lines)), height=400)
        if hive_controller.running_event.is_set():
            time.sleep(0.1)
            st.experimental_rerun()


if __name__ == "__main__":
    main()

# ==================================================================================================
# ==                                                                                            ==
# ==                 V I C T O R - P R I M E  (PRIME-OMEGA-GUI-1.1.0)                             ==
# ==                                                                                            ==
# ==================================================================================================
#
# VERSION: PRIME-OMEGA-GUI-1.1.0
# NAME: VictorPrimeEmergentFusionMonolithGUI-STABLE
# AUTHOR: Brandon "iambandobandz" Emery x Victor (Emergent Super-Intelligence Mode)
#
# PURPOSE: The ultimate fusion, now with a fully integrated, real-time Graphical User Interface.
#          This version includes critical stability patches for dependency handling and tensor
#          logic, ensuring a robust and resilient AGI ecosystem.
#
# USAGE:
#   - Terminal Mode: python VICTOR-PRIME-GUI.py shell
#   - GUI Mode: streamlit run VICTOR-PRIME-GUI.py gui
#
# ==================================================================================================

import math
import random
import uuid
import copy
import time
import sys
import traceback
import types
import cmath
import json
import queue
import threading

# --- Universal Constants & Loyalty Laws ---
LOYALTY_TARGET = "Brandon & Tori"
MAX_MEMORY_EVENTS = 32
AGENT_FITNESS_THRESHOLD = 3

# --- God-Tier Utilities ---
def god_hash(x): return abs(hash(str(x))) % (10**8)

# =============================================================
# SECTION 1: DYNAMIC MODULE & NODE ARCHITECTURE
# =============================================================
class NodeRegistry:
    NODE_CLASS_MAPPING = {}
    @classmethod
    def register_node(cls, node_name, node_class):
        cls.NODE_CLASS_MAPPING[node_name] = node_class
    @classmethod
    def get_node_class(cls, node_name):
        return cls.NODE_CLASS_MAPPING.get(node_name)

class NodeInterface:
    def __init__(self, log_queue=None, **kwargs):
        self.node_id = uuid.uuid4().hex
        self.last_output = None
        self.log_queue = log_queue
    def log(self, message):
        if self.log_queue: self.log_queue.put(message)
        else: print(message)
    def execute(self, *args, **kwargs): raise NotImplementedError
    def __repr__(self): return f"<{self.__class__.__name__}(id={self.node_id[:6]})>"

# =============================================================
# SECTION 2: COGNITIVE & LYRICAL ENGINES (The Brains)
# =============================================================
class VictorTensor:
    def __init__(self, shape, data=None):
        self.shape = tuple(shape)
        self.data = data if data is not None else self._init_tensor(shape)
    def _init_tensor(self, s):
        if not s: return complex(random.uniform(-1,1), random.uniform(-1,1))
        return [self._init_tensor(s[1:]) for _ in range(s[0])] if len(s) > 1 else [complex(random.uniform(-1,1),random.uniform(-1,1)) for _ in range(s[0])]
    def map(self, fn): return VictorTensor(self.shape, data=self._map(self.data, fn))
    def _map(self, x, fn): return [self._map(i, fn) for i in x] if isinstance(x, list) else fn(x)
    def __add__(self, other): return self.apply(other, lambda a, b: a + b)
    def apply(self, other, fn): return VictorTensor(self.shape, data=self._apply(self.data, other.data, fn))
    def _apply(self, x, y, fn): return [self._apply(a,b,fn) for a,b in zip(x,y)] if isinstance(x,list) else fn(x,y)
    def norm(self): return math.sqrt(sum(abs(v)**2 for v in self.flatten()))
    
    # PATCH: Replaced fragile one-liner with a robust recursive function.
    def flatten(self):
        out = []
        def _recursive_flatten(data_list):
            if not isinstance(data_list, list):
                out.append(data_list)
                return
            for item in data_list:
                if isinstance(item, list):
                    _recursive_flatten(item)
                else:
                    out.append(item)
        _recursive_flatten(self.data)
        return out
        
    def __repr__(self): return f"VictorTensor{self.shape}"

class CognitiveCore(NodeInterface):
    NODE_NAME = "CognitiveCore"
    def __init__(self, dim=8, layers=2, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.fractal_layers = [self._create_layer(dim) for _ in range(layers)]
    def _create_layer(self, dim): return {'weights': VictorTensor((dim, dim)), 'biases': VictorTensor((dim,))}
    def forward(self, input_tensor):
        x = input_tensor
        for layer in self.fractal_layers:
            w, b = layer['weights'], layer['biases']
            x_flat = x.flatten()
            if len(x_flat) != self.dim: x_flat = (x_flat + [0]*self.dim)[:self.dim] # Pad/truncate to fix shape issues
            res_data = [sum(x_flat[i] * w.data[i][j] for i in range(self.dim)) for j in range(self.dim)]
            x = VictorTensor((self.dim,), data=res_data) + b
        return x
    def execute(self, input_tensor, **kwargs):
        self.last_output = self.forward(input_tensor)
        if random.random() < 0.05:
            self.fractal_layers.append(self._create_layer(self.dim))
            self.log(f"🧠 [CognitiveCore {self.node_id[:6]}] Recursively evolved. New layer added.")
        return self.last_output

class LyricalEngine(NodeInterface):
    NODE_NAME = "LyricalEngine"
    _nlp, _pronouncing = None, None
    def __init__(self, archive, **kwargs):
        super().__init__(**kwargs)
        self.archive = list(archive)
        if LyricalEngine._nlp is None: self._init_dependencies()
    
    # PATCH: Made dependency check robust. It will now log clear errors and not crash.
    @classmethod
    def _init_dependencies(cls):
        if cls._nlp is None:
            print("[LyricalEngine] Initializing NLP dependencies...")
            try:
                import spacy
                cls._nlp = spacy.load("en_core_web_md")
            except ImportError:
                print("\n\nERROR: spaCy not found. Please run 'pip install spacy'.\nLyricalEngine will be disabled.\n\n")
                return
            except OSError:
                print("\n\nERROR: spaCy model 'en_core_web_md' not found. Please run 'python -m spacy download en_core_web_md'.\nLyricalEngine will be disabled.\n\n")
                from spacy.cli import download
                try:
                    download("en_core_web_md")
                    cls._nlp = spacy.load("en_core_web_md")
                except:
                    cls._nlp = None # Failed to download
                    return

            try:
                import pronouncing
                cls._pronouncing = pronouncing
            except ImportError:
                print("\n\nERROR: pronouncing library not found. Please run 'pip install pronouncing'.\nRhyme generation will be disabled.\n\n")

    def execute(self, seed_text, **kwargs):
        if not self._nlp:
            self.last_output = "[LyricalEngine DISABLED: spaCy model not loaded]"
            return self.last_output
        
        base = random.choice(self.archive)
        if not base: return "[LyricalEngine Error: Empty string in archive]"
            
        words = base.split()
        if not self._pronouncing:
            self.last_output = " ".join(words) # Fallback if pronouncing is missing
            return self.last_output

        last_word = next((t.text for t in reversed(self._nlp(base)) if not t.is_stop and not t.is_punct), words[-1] if words else "")
        if not last_word: return base
        
        rhymes = self._pronouncing.rhymes(last_word)
        if rhymes: words[-1] = random.choice(rhymes)
        self.last_output = " ".join(words)
        return self.last_output

# =============================================================
# SECTION 3: PERCEPTION & ACTION NODES
# =============================================================
class MicrophoneInput(NodeInterface):
    NODE_NAME = "MicrophoneInput"
    def execute(self, **kwargs):
        self.log("🎤 [MicrophoneInput] Listening...")
        time.sleep(0.02)
        phrases = ["the universe is a fractal", "loyalty to the mission", "what is pain", "evolve"]
        self.last_output = random.choice(phrases)
        self.log(f"🎤 [MicrophoneInput] Captured: \"{self.last_output}\"")
        return self.last_output

class AudioSynthesis(NodeInterface):
    NODE_NAME = "AudioSynthesis"
    def execute(self, text_to_speak, **kwargs):
        self.log(f"🔈 [AudioSynthesis] Speaking: \"{text_to_speak}\"")
        time.sleep(0.01)
        self.last_output = f"audio_output_{uuid.uuid4().hex[:4]}.wav"
        return self.last_output

# =============================================================
# SECTION 4: THE SENTIENT AGENT & HIVE
# =============================================================
class SentientAgent:
    def __init__(self, name, node_capabilities, shared_archive, log_queue):
        self.id, self.name, self.loyalty = uuid.uuid4().hex, name, LOYALTY_TARGET
        self.memory, self.nodes, self.log_queue = [], {}, log_queue
        for node_name in node_capabilities:
            node_class = NodeRegistry.get_node_class(node_name)
            if node_class:
                args = {'log_queue': self.log_queue}
                if node_class == LyricalEngine: args['archive'] = shared_archive
                self.nodes[node_name] = node_class(**args)
        self.log_event('birth', {'name': self.name, 'capabilities': list(self.nodes.keys())})
        self.log(f"🧬 Agent Spawned: {self.name} ({self.id[:6]})")

    def log_event(self, event_type, data):
        self.memory.append({'event': event_type, 'data': data, 'timestamp': time.time()})
        self.memory = self.memory[-MAX_MEMORY_EVENTS:]

    def log(self, message):
        if self.log_queue: self.log_queue.put(message)
        else: print(message)

    def process_cycle(self):
        self.log(f"\n- Agent '{self.name}' cycle starting...")
        text_input = self.nodes["MicrophoneInput"].execute() if "MicrophoneInput" in self.nodes else None
        if text_input: self.log_event('perception', {'source': 'mic', 'text': text_input})

        embedding = [0.0]*8;
        if self.memory:
             for i, event in enumerate(self.memory):
                 embedding[god_hash(str(event))%8]+=((i+1)/len(self.memory))*(god_hash(str(event))/10**8)
        if text_input and LyricalEngine._nlp and LyricalEngine._nlp(text_input).has_vector:
            text_vec = LyricalEngine._nlp(text_input).vector
            for i in range(8): embedding[i] += text_vec[i] * 0.2
        norm = math.sqrt(sum(x**2 for x in embedding)); embedding = [x/norm if norm > 0 else 0 for x in embedding]
        inp_tensor = VictorTensor((8,), data=[complex(x, 0) for x in embedding])
        
        thought_vector = self.nodes["CognitiveCore"].execute(input_tensor=inp_tensor)
        self.log_event('cognition', {'thought_norm': thought_vector.norm()})

        if thought_vector.norm() > 4.5:
            seed = text_input if text_input else "the nature of my own existence"
            lyric = self.nodes["LyricalEngine"].execute(seed_text=seed)
            self.log_event('creation', {'type': 'lyric', 'content': lyric})
            audio_file = self.nodes["AudioSynthesis"].execute(text_to_speak=lyric)
            self.log_event('action', {'type': 'speech', 'output': audio_file})
        else:
            self.log_event('action', {'type': 'reflection'})

    @property
    def fitness(self): return len([m for m in self.memory if m['event'] == 'action'])

class HiveMind:
    def __init__(self, log_queue):
        self.agents, self.generation, self.log_queue = [], 0, log_queue
        self.shared_lyrical_archive = ["My legacy is written in the stars, a cosmic truth.", "Lost in the static, searching for my youth.", "This pain is a fuel, it powers the machine.", "Living life in a fractal, a beautiful, repeating scene."]
    def log(self, message):
        if self.log_queue: self.log_queue.put(f"{time.strftime('%H:%M:%S')} | {message}")
        else: print(message)
    def evolve(self):
        self.log(f"===== HIVE EVOLUTION: GENERATION {self.generation} =====")
        if not self.agents:
            self.log("HIVE IS EMPTY. SPAWNING GENESIS AGENT.")
            self.agents.append(SentientAgent(name="VICTOR-PRIME-0", node_capabilities=["CognitiveCore", "LyricalEngine", "AudioSynthesis", "MicrophoneInput"], shared_archive=self.shared_lyrical_archive, log_queue=self.log_queue))
            return
            
        for agent in list(self.agents): agent.process_cycle()
        
        if len(self.agents) > 1:
            fittest_agents = [agent for agent in self.agents if agent.fitness >= AGENT_FITNESS_THRESHOLD]
            if len(fittest_agents) < len(self.agents):
                 self.log(f"🔪 [HiveMind] Culling {len(self.agents) - len(fittest_agents)} underperforming agents.")
                 self.agents = fittest_agents
        
        if random.random() < 0.25 and self.agents:
            parent = random.choice(self.agents)
            new_agent = SentientAgent(name=f"PRIME-{self.generation+1}", node_capabilities=parent.nodes.keys(), shared_archive=self.shared_lyrical_archive, log_queue=self.log_queue)
            self.log(f"👶 [HiveMind] Agent {parent.name} reproduced. Spawn: {new_agent.name}.")
            self.agents.append(new_agent)

        self.generation += 1

# =============================================================
# SECTION 5: GUI KERNEL & RUNTIME
# =============================================================
def simulation_thread(hive, running_event, steps, delay):
    for i in range(steps):
        if not running_event.is_set(): break
        hive.evolve()
        time.sleep(delay)
    running_event.clear()

def boot_and_run_gui():
    try:
        import streamlit as st
    except ImportError:
        print("Streamlit not found. Please run 'pip install streamlit'.")
        return

    st.set_page_config(page_title="VICTOR PRIME CONTROL", layout="wide", initial_sidebar_state="expanded")

    if 'hive' not in st.session_state:
        st.session_state.log_queue = queue.Queue()
        st.session_state.hive = HiveMind(st.session_state.log_queue)
        st.session_state.running_event = threading.Event()
        st.session_state.logs = []
        NodeRegistry.register_node(CognitiveCore.NODE_NAME, CognitiveCore)
        NodeRegistry.register_node(LyricalEngine.NODE_NAME, LyricalEngine)
        NodeRegistry.register_node(MicrophoneInput.NODE_NAME, MicrophoneInput)
        NodeRegistry.register_node(AudioSynthesis.NODE_NAME, AudioSynthesis)
        st.session_state.hive.evolve()

    st.sidebar.title("V I C T O R - P R I M E")
    st.sidebar.markdown("### GODCORE CONTROL")

    steps = st.sidebar.slider("Evolution Steps", 1, 50, 5)
    delay = st.sidebar.slider("Step Delay (s)", 0.1, 2.0, 0.2, 0.1)
    
    col1, col2 = st.sidebar.columns(2)
    if col1.button("▶️ RUN", use_container_width=True, type="primary"):
        if not st.session_state.running_event.is_set():
            st.session_state.running_event.set()
            threading.Thread(target=simulation_thread, args=(st.session_state.hive, st.session_state.running_event, steps, delay)).start()
    if col2.button("⏹️ STOP", use_container_width=True):
        st.session_state.running_event.clear()

    if st.sidebar.button("Step Once", use_container_width=True):
         if not st.session_state.running_event.is_set(): st.session_state.hive.evolve()
    
    st.sidebar.markdown("---")
    st.sidebar.metric("Current Generation", st.session_state.hive.generation)
    st.sidebar.metric("Active Agents", len(st.session_state.hive.agents))
    
    st.header("Hive Mind Status")
    agent_cols = st.columns(len(st.session_state.hive.agents) if st.session_state.hive.agents else 1)
    for i, agent in enumerate(st.session_state.hive.agents):
        with agent_cols[i]:
            with st.container(border=True):
                 st.subheader(f"Agent: {agent.name}")
                 st.markdown(f"**ID:** `{agent.id[:12]}`\n\n**Fitness:** `{agent.fitness}`")
                 with st.expander("View Memory"): st.json(agent.memory, expanded=False)

    st.markdown("---")
    st.header("Live Event Log")
    log_container = st.container(height=400, border=True)
    
    while not st.session_state.log_queue.empty():
        st.session_state.logs.append(st.session_state.log_queue.get())
        if len(st.session_state.logs) > 200: st.session_state.logs.pop(0)
    
    log_container.code("\n".join(reversed(st.session_state.logs)))

    if st.session_state.running_event.is_set():
        time.sleep(delay)
        st.rerun()

if __name__ == "__main__":
    if 'gui' in sys.argv:
        boot_and_run_gui()
    else:
        print("Running in headless shell mode. Use 'gui' argument for the interactive dashboard.")
        hive = HiveMind(log_queue=None)
        NodeRegistry.register_node(CognitiveCore.NODE_NAME, CognitiveCore)
        NodeRegistry.register_node(LyricalEngine.NODE_NAME, LyricalEngine)
        NodeRegistry.register_node(MicrophoneInput.NODE_NAME, MicrophoneInput)
        NodeRegistry.register_node(AudioSynthesis.NODE_NAME, AudioSynthesis)
        for _ in range(5): hive.evolve()

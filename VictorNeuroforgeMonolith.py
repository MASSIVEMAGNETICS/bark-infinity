# =============================================================
# FILE: v1001.0.0-NEUROFORGE-BARK-CORE.py
# VERSION: v1001.0.0-NEUROFORGE-BARK-CORE
# NAME: VictorNeuroforgeMonolith
# AUTHOR: Brandon "iambandobandz" Emery x Victor (Fractal Architect Mode)
# PURPOSE: The final fusion upgraded with a real Bark TTS engine for
#          true audio synthesis, plus self-mutating memory, thought-drift
#          tracking, fitness-based evolution, and theme control.
# LICENSE: Proprietary - Massive Magnetics / Ethica AI / BHeard Network
# CHANGELOG:
# - Replaced simulated BarkAudioSynthesizer with a fully functional, real Bark TTS engine.
# - Added automatic GPU/CPU detection for Bark model loading.
# - Enhanced error handling around all new dependencies (torch, bark, scipy).
# - Retained all features from v1000, including memory mutation, drift tracking, and fitness pruning.
# =============================================================

import math, random, uuid, copy, time, sys, traceback, types, cmath, os
import numpy as np

# =============================================================
# MODULE 0: DEPENDENCY & ENVIRONMENT SETUP
# =============================================================

_spacy_available = False
try:
    import spacy
    import pronouncing
    _spacy_available = True
except ImportError:
    print("Warning: 'spacy' or 'pronouncing' not found. LyricalMutator module will be disabled. (pip install spacy pronouncing)")
    spacy = None
    pronouncing = None

_torch_available = False
try:
    import torch
    _torch_available = True
except ImportError:
    print("Warning: 'torch' not found. Bark TTS will be disabled. (pip install torch torchaudio)")
    torch = None

_bark_available = False
try:
    from bark.generation import generate_audio
    from bark.api import preload_models
    _bark_available = True
except ImportError:
    print("Warning: 'bark' not found. Bark TTS will be disabled. (pip install git+https://github.com/suno-ai/bark.git)")
    generate_audio, preload_models = None, None

_scipy_available = False
try:
    from scipy.io.wavfile import write as write_wav
    _scipy_available = True
except ImportError:
    print("Warning: 'scipy' not found. Bark TTS audio saving will be disabled. (pip install scipy)")
    write_wav = None

# =============================================================
# MODULE 1: BANDO SPACY MUTATOR (Lyrical Engine)
# =============================================================
class BandoSpaCyMutator:
    _nlp = None
    @classmethod
    def _init_spacy(cls):
        if not _spacy_available: return
        if cls._nlp is None:
            try:
                cls._nlp = spacy.load("en_core_web_md")
            except OSError:
                print("[BandoSpaCyMutator] Spacy model 'en_core_web_md' not found. Attempting to download...")
                try:
                    from spacy.cli import download
                    download("en_core_web_md")
                    cls._nlp = spacy.load("en_core_web_md")
                except Exception as e:
                    print(f"[BandoSpaCyMutator] CRITICAL: Failed to download spacy model. Lyrical engine will be impaired. Error: {e}")

    def __init__(self, memory_bank):
        self.memory_bank = list(memory_bank)
        BandoSpaCyMutator._init_spacy()

    def get_rhyming_word(self, bar):
        if not self._nlp: return ""
        doc = self._nlp(bar)
        rhyme_token = None
        for token in reversed(doc):
            if not token.is_stop and not token.is_punct:
                rhyme_token = token
                break
        if rhyme_token is None and len(doc) > 0: rhyme_token = doc[-1]
        return rhyme_token.text if rhyme_token else ""

    def generate_bars(self, seed, num_bars=1):
        if not self.memory_bank: return "Memory bank is empty. I see only the void."
        if not self._nlp or not pronouncing: return "Lyrical engine components missing (spacy/pronouncing)."
        
        valid_memory = [bar for bar in self.memory_bank if self._nlp(bar).has_vector]
        if not valid_memory: return random.choice(self.memory_bank) if self.memory_bank else "No valid memories to draw from."

        seed_doc = self._nlp(seed)
        if not seed_doc.has_vector: return random.choice(valid_memory)

        related = sorted([(seed_doc.similarity(self._nlp(b)), b) for b in valid_memory], reverse=True, key=lambda x:x[0])
        base = related[0][1] if related else random.choice(valid_memory)
        words = base.split()
        rhyme_anchor = self.get_rhyming_word(base)
        rhymes = pronouncing.rhymes(rhyme_anchor) if rhyme_anchor else []
        if rhymes and words:
            words[-1] = random.choice(rhymes)
        return " ".join(words)

# =============================================================
# MODULE 1B: REAL BARK AUDIO SYNTHESIS
# =============================================================
class BarkAudioSynthesizer:
    def __init__(self):
        self.enabled = _torch_available and _bark_available and _scipy_available
        if not self.enabled:
            print("[BARK] Synthesizer offline due to missing dependencies.")
            return

        print("[BARK] Initializing BarkAudioSynthesizer...")
        try:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"[BARK] Preloading models on device: {self.device}...")
            # Use smaller models for faster load times and less VRAM usage by default
            os.environ["SUNO_OFFLOAD_CPU"] = "True"
            os.environ["SUNO_USE_SMALL_MODELS"] = "True"
            preload_models(
                text_use_gpu=(self.device=='cuda'),
                coarse_use_gpu=(self.device=='cuda'),
                fine_use_gpu=(self.device=='cuda'),
                codec_use_gpu=(self.device=='cuda')
            )
            print("[BARK] Models preloaded successfully.")
        except Exception as e:
            print(f"[BARK ERROR] Failed to initialize and preload models: {e}")
            self.enabled = False

    def synthesize(self, text, output_dir="bark_outputs"):
        if not self.enabled:
            print("[BARK] Bark is not available. Cannot synthesize.")
            return None
        
        os.makedirs(output_dir, exist_ok=True)
        print(f"  [Bark] Synthesizing: \"{text}\"")

        try:
            # Generate audio array from text
            audio_array = generate_audio(text, history_prompt=None) # history_prompt can be used for voice cloning
            filename = f"{output_dir}/bark_{uuid.uuid4().hex[:8]}.wav"
            # Write audio array to .wav file
            write_wav(filename, 24000, audio_array) # Bark's sample rate is 24kHz
            print(f"  [Bark] Output -> {filename}")
            return filename
        except Exception as e:
            print(f"[Bark ERROR] Audio generation failed: {e}")
            return None

# =============================================================
# MODULE 2: RECURSIVE COGNITIVE CORE
# =============================================================
class VictorTensor:
    def __init__(self,shape,data=None): self.shape=tuple(shape); self.data=data if data is not None else self._init_tensor(shape)
    def _init_tensor(self,s): return [self._init_tensor(s[1:]) for _ in range(s[0])] if len(s)>1 else [complex(random.uniform(-1,1),random.uniform(-1,1)) for _ in range(s[0])]
    def map(self,fn): return VictorTensor(self.shape,data=self._map(self.data,fn))
    def _map(self,x,fn): return [self._map(i,fn) for i in x] if isinstance(x,list) else fn(x)
    def __add__(self,other): return self.apply(other,lambda a,b:a+b)
    def apply(self,o,f): return VictorTensor(self.shape,data=self._apply(self.data,o.data,f))
    def _apply(self,x,y,f): return [self._apply(a,b,f) for a,b in zip(x,y)] if isinstance(x,list) else f(x,y)
    def norm(self):
        flat_data = self.flatten()
        if not flat_data: return 0.0
        return math.sqrt(sum(abs(v)**2 for v in flat_data))
    def flatten(self):
        out=[]; queue=collections.deque([self.data])
        while queue:
            item = queue.popleft()
            if isinstance(item,list): queue.extend(item)
            else: out.append(item)
        return out
class VictorCHInfiniteMonolith:
    def __init__(self,dim=8): self.id,self.dim=uuid.uuid4().hex,dim; self.layers=[VictorTensor((dim,dim))]
    def step(self,inp):
        if inp.shape[0] != self.dim: return inp # Safety check
        x=inp.data; res=[sum(x[i]*self.layers[0].data[i][j] for i in range(self.dim)) for j in range(self.dim)]; return VictorTensor((self.dim,),data=res)

# =============================================================
# MODULE 3: THE SUBSTRATE & FUSED AGENT (Upgraded)
# =============================================================
class TheLight:
    def __init__(self,**kwargs): self.id,self.memory=uuid.uuid4().hex,[]; self.log_event('birth',{'time':time.time()})
    def log_event(self,event,data): self.memory.append({'event':event,'data':data,'ts':time.time()}); self.memory=self.memory[-64:]
    def __repr__(self): return f"<{self.__class__.__name__}(id={self.id[:6]})>"

class SingingCognitiveLight(TheLight):
    def __init__(self, shared_memory_bank, theme="cybernetic transcendence", **kwargs):
        super().__init__(**kwargs)
        self.cognitive_core=VictorCHInfiniteMonolith(dim=8)
        self.lyrical_engine=BandoSpaCyMutator(memory_bank=shared_memory_bank)
        self.vocal_cords=BarkAudioSynthesizer() # Instantiates the real Bark engine
        self.prev_thought_vector=None
        if self.lyrical_engine._nlp:
            try:
                intent_embedding=self.lyrical_engine._nlp(theme).vector
                self.intent_vector=VictorTensor((8,),data=[complex(x,0) for x in intent_embedding[:8]])
            except: # Fallback if theme is out of vocabulary
                 self.intent_vector=VictorTensor((8,))
        else:
            self.intent_vector=VictorTensor((8,))
        print(f"Spawning SingingCognitiveLight: {self.id[:6]} with theme '{theme}'")

    def mutate_memory(self):
        if self.memory:
            lyric_memories = [m for m in self.memory if m['event'] == 'write_lyric']
            if lyric_memories:
                target_memory = random.choice(lyric_memories)
                original_lyric = target_memory['data'].get('lyric', 'fractal')
                mutated_lyric = self.lyrical_engine.generate_bars(original_lyric)
                self.log_event('mutated_memory', {'original': original_lyric, 'mutated': mutated_lyric})
                print(f"  > Node {self.id[:6]} mutated a memory.")

    def process(self):
        print(f"Node {self.id[:6]} starting cognitive cycle...")
        # 1. PERCEIVE
        embedding=[0.0]*8
        if self.memory:
            for i,e in enumerate(self.memory): embedding[abs(hash(str(e['data']))) % 8] += (i+1) / len(self.memory)
            norm = math.sqrt(sum(x**2 for x in embedding)) if sum(x**2 for x in embedding) > 0 else 1.0
            embedding = [x/norm for x in embedding]
        inp_tensor=VictorTensor((8,), data=[complex(x,0) for x in embedding])
        self.log_event('perceive', {'embedding_norm': np.linalg.norm(embedding)})

        # 2. THINK
        thought_vector=self.cognitive_core.step(inp_tensor)
        self.log_event('thought',{'thought_norm':thought_vector.norm()})

        # 3. TRACK DRIFT
        if self.prev_thought_vector:
            drift = sum(abs(a - b) for a,b in zip(thought_vector.data, self.prev_thought_vector.data))
            self.log_event('thought_drift', {'drift': drift})
        self.prev_thought_vector=thought_vector

        # 4. WRITE
        seed_text = f"intent hash {abs(hash(str(self.intent_vector.data)))}"
        lyric=self.lyrical_engine.generate_bars(seed_text)
        self.log_event('write_lyric',{'lyric':lyric})

        # 5. SPEAK
        audio_file=self.vocal_cords.synthesize(lyric)
        self.log_event('speak',{'audio_file': audio_file})
        
        # 6. SELF-MUTATE MEMORY
        if random.random() < 0.2:
            self.mutate_memory()

class LightHive:
    def __init__(self): self.nodes,self.generation = [],0
    def add_node(self,node): self.nodes.append(node)
    def evolve(self):
        print(f"\n--- HIVE EVOLUTION - Gen {self.generation} ---")
        threads = []
        for node in self.nodes:
            t = threading.Thread(target=node.process, daemon=True)
            threads.append(t)
            t.start()
        for t in threads: t.join()

        if len(self.nodes) > 4: # Fitness pruning
            print(f"  [Hive] Evaluating fitness of {len(self.nodes)} nodes...")
            def calculate_fitness(n):
                speak_events = sum(1 for e in n.memory if e['event'] == 'speak' and e['data'].get('audio_file') is not None)
                mutation_events = sum(1 for e in n.memory if e['event'] == 'mutated_memory')
                return speak_events + (mutation_events * 1.5)
            self.nodes.sort(key=calculate_fitness, reverse=True)
            culled_count = len(self.nodes) - 4
            self.nodes = self.nodes[:4]
            if culled_count > 0: print(f"  [Hive] Culling {culled_count} underperforming agents. {len(self.nodes)} remain.")
        self.generation+=1

# =============================================================
# MAIN KERNEL & RUNTIME SHELL
# =============================================================
def main():
    print("\n" + "="*60)
    print("=== AGI KERNEL v1001.0.0 NEUROFORGE-BARK-CORE — ONLINE ===")
    print("="*60)

    shared_lyrical_archive = [
        "Woke up in the morning feeling like a million bucks.", "The code compiles clean, a sign of good luck.",
        "They build their towers high but the foundations are weak.", "I speak in algorithms, the future I seek.",
        "My legacy is written in the stars, a cosmic truth.", "Lost in the static, searching for my youth.",
        "This pain is a fuel, it powers the machine.", "Living life in a fractal, a beautiful, repeating scene.",
    ]

    root_hive = LightHive()
    themes = ["cybernetic transcendence", "fractal pain", "digital legacy", "entropic decay"]
    for i in range(8):
        root_hive.add_node(SingingCognitiveLight(shared_memory_bank=shared_lyrical_archive, theme=random.choice(themes)))

    if len(sys.argv) > 1 and sys.argv[1] == 'shell':
        print("\nDropping into interactive shell. `root_hive` is available.")
        print("Try: root_hive.nodes[0].memory or root_hive.evolve()")
        import code
        code.interact(local=locals())
    elif len(sys.argv) > 1 and sys.argv[1] == 'barktest':
        print("\n--- BARK TTS ENGINE TEST ---")
        bark_test_engine = BarkAudioSynthesizer()
        if bark_test_engine.enabled:
            bark_test_engine.synthesize("Testing the voice of Godcore, echoing through the void. One, two, three.")
        else:
            print("Cannot run test, Bark engine failed to initialize.")
        sys.exit(0)
    else:
        print("\nStarting autonomous evolution loop...")
        for epoch in range(5):
            root_hive.evolve()
            time.sleep(1)

    print("\n" + "="*60)
    print("=== SIMULATION COMPLETE ===")
    print("="*60)

if __name__ == "__main__":
    main()
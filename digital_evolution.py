# -*- coding: utf-8 -*-
"""Digital evolution and cognitive swarm module."""

import uuid
import random
import copy
from collections import Counter


class LightMind:
    """Simple cognitive engine with memory and decision bias."""

    def __init__(self, dna):
        self.memory = []
        self.memory_capacity = dna.get('memory_capacity', 10)
        self.learning_rate = dna.get('learning_rate', 0.05)
        self.strategy_bias = dna.get('strategy_bias', 'explore')

    def learn(self, stimulus):
        reward = self.evaluate(stimulus)
        self.memory.append((stimulus, reward))
        if len(self.memory) > self.memory_capacity:
            self.memory.pop(0)

    def evaluate(self, stimulus):
        if stimulus == 'resource':
            return +1 * self.learning_rate
        if stimulus == 'threat':
            return -1 * self.learning_rate
        if stimulus == 'ally':
            return +0.5 * self.learning_rate
        return 0

    def decide_action(self):
        return self.strategy_bias


class TheLight:
    """Node with heritable DNA and simple behavior."""

    STATES = {'field', 'plasma', 'solid'}

    def __init__(self, quantization=1.0, state='field', dimensions=3,
                 radius=1.0, entropy=0.01, temperature=0.5,
                 id=None, dna=None, generation=0, parent_id=None,
                 genealogy=None):
        self.id = id or str(uuid.uuid4())
        self.parent_id = parent_id
        self.generation = generation
        self.genealogy = list(genealogy) if genealogy else []
        if self.id not in self.genealogy:
            self.genealogy.append(self.id)

        self.quantization = quantization
        self.state = state if state in self.STATES else 'field'
        self.dimensions = dimensions
        self.radius = radius
        self.entropy = entropy
        self.temperature = temperature

        self.dna = dna or self.generate_dna()
        self.express_phenotype()
        self.mind = LightMind(self.dna)
        self.morph_history = []
        self.perimeter_points = []

    def generate_dna(self):
        return {
            'q': self.quantization,
            'dims': self.dimensions,
            'entropy': self.entropy,
            'temp': self.temperature,
            'mutation_rate': 0.07,
            'bloom_factor': random.randint(2, 5),
            'aggression': random.uniform(0.0, 1.0),
            'memory_capacity': random.randint(5, 20),
            'learning_rate': random.uniform(0.01, 0.1),
            'strategy_bias': random.choice(['explore', 'exploit', 'conserve']),
        }

    def express_phenotype(self):
        self.color = self.dna.get('color', random.choice(['white', 'blue', 'gold']))
        self.bloom_rate = self.dna.get('bloom_factor', 1)
        self.aggression = self.dna.get('aggression', 0.1)
        self.memory_capacity = self.dna.get('memory_capacity', 10)
        self.learning_rate = self.dna.get('learning_rate', 0.05)
        self.strategy_bias = self.dna.get('strategy_bias', 'explore')

    def mutate_dna(self, dna=None):
        d = copy.deepcopy(dna or self.dna)
        for key in list(d.keys()):
            if key in ['q', 'entropy', 'temp']:
                if random.random() < d['mutation_rate']:
                    d[key] *= random.uniform(0.95, 1.07)
            if key == 'dims':
                if random.random() < d['mutation_rate']:
                    d[key] = max(2, min(12, d[key] + random.choice([-1, 1])))
            if key == 'bloom_factor':
                if random.random() < d['mutation_rate']:
                    d[key] = max(2, min(10, d[key] + random.choice([-1, 1])))
            if key in ['aggression', 'learning_rate']:
                if random.random() < d['mutation_rate']:
                    d[key] = max(0.0, min(1.0, d[key] + random.uniform(-0.1, 0.1)))
            if key == 'strategy_bias':
                if random.random() < d['mutation_rate']:
                    d[key] = random.choice(['explore', 'exploit', 'conserve'])
        return d

    def act(self, environment=None):
        stimulus = random.choice(['resource', 'threat', 'ally'])
        self.mind.learn(stimulus)
        action = self.mind.decide_action()
        if action == 'explore':
            self.entropy += 0.002
        elif action == 'exploit':
            self.entropy = max(0.0, self.entropy - 0.003)
        elif action == 'conserve':
            self.entropy = max(0.0, self.entropy - 0.001)

    def coherence_score(self):
        # Simplified placeholder coherence calculation
        return max(0.0, 1.0 - self.entropy)

    def spawn_shard(self, min_q=0.3, coherence_threshold=0.97):
        if self.coherence_score() >= coherence_threshold:
            new_dna = self.mutate_dna()
            shard = TheLight(
                quantization=max(new_dna['q'] * 0.5, min_q),
                state=self.state,
                dimensions=new_dna['dims'],
                radius=self.radius * 0.9,
                entropy=max(new_dna['entropy'] * 0.5, 0.001),
                temperature=new_dna['temp'],
                id=str(uuid.uuid4()),
                dna=new_dna,
                generation=self.generation + 1,
                parent_id=self.id,
                genealogy=self.genealogy + [self.id]
            )
            return shard
        return None

    def info(self):
        return {
            'id': self.id,
            'generation': self.generation,
            'state': self.state,
            'quantization': self.quantization,
            'dimensions': self.dimensions,
            'radius': self.radius,
            'entropy': self.entropy,
            'temperature': self.temperature,
            'coherence': self.coherence_score(),
            'dna': self.dna,
            'genealogy': self.genealogy,
        }


class LightHive:
    """Collection of nodes with evolutionary selection."""

    def __init__(self):
        self.nodes = []
        self.generation = 0

    def add_node(self, node):
        self.nodes.append(node)

    def evolve(self):
        for node in list(self.nodes):
            node.act(self.nodes)
            shard = node.spawn_shard()
            if shard is not None:
                self.nodes.append(shard)
        self.evolutionary_selection()
        self.generation += 1

    def evolutionary_selection(self, top_fraction=0.5, min_population=4):
        scored = sorted(
            self.nodes,
            key=lambda n: (1 / (n.entropy + 1e-5)) + n.bloom_rate,
            reverse=True
        )
        survivors = scored[:max(min_population, int(top_fraction * len(scored)))]
        self.nodes = list(survivors)

    def cognitive_distribution(self, trait='strategy_bias'):
        return Counter(n.dna.get(trait, 'unknown') for n in self.nodes)

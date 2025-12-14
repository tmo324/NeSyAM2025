"""
VIPER - decision tree trained on expert demonstrations
"""
import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import accuracy_score
import pickle


class VIPERAgent:
    def __init__(self, max_depth=10, min_samples_split=10, min_samples_leaf=5, random_state=42):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        self.tree = None
        self.training_accuracy = None

    def train(self, states, actions):
        print(f"training on {len(states)} samples...")

        self.tree = DecisionTreeClassifier(
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            random_state=self.random_state
        )
        self.tree.fit(states, actions)

        preds = self.tree.predict(states)
        self.training_accuracy = accuracy_score(actions, preds)

        print(f"  acc={self.training_accuracy:.4f}, depth={self.tree.get_depth()}, leaves={self.tree.get_n_leaves()}")

    def predict(self, state):
        if self.tree is None:
            raise ValueError("not trained")
        if state.ndim == 1:
            state = state.reshape(1, -1)
        return int(self.tree.predict(state)[0])

    def get_depth(self):
        return self.tree.get_depth() if self.tree else 0

    def get_num_leaves(self):
        return self.tree.get_n_leaves() if self.tree else 0

    def get_tree_text(self, feature_names=None):
        if self.tree is None:
            return "not trained"
        return export_text(self.tree, feature_names=feature_names, show_weights=True)

    def save(self, path):
        data = {
            'max_depth': self.max_depth,
            'min_samples_split': self.min_samples_split,
            'min_samples_leaf': self.min_samples_leaf,
            'random_state': self.random_state,
            'tree': self.tree,
            'training_accuracy': self.training_accuracy
        }
        with open(path, 'wb') as f:
            pickle.dump(data, f)
        print(f"saved to {path}")

    @staticmethod
    def load(path):
        with open(path, 'rb') as f:
            data = pickle.load(f)

        agent = VIPERAgent(
            max_depth=data['max_depth'],
            min_samples_split=data['min_samples_split'],
            min_samples_leaf=data['min_samples_leaf'],
            random_state=data['random_state']
        )
        agent.tree = data['tree']
        agent.training_accuracy = data['training_accuracy']

        print(f"loaded from {path}")
        print(f"  depth={agent.get_depth()}, leaves={agent.get_num_leaves()}, acc={agent.training_accuracy:.4f}")
        return agent

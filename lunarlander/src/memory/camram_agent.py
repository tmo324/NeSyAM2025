"""
CAM-RAM - k-NN episodic memory for RL
"""
import numpy as np
import pickle


class CAMRAM:
    def __init__(self, num_actions, k=5, max_mem=10000, epsilon=0.1,
                 eps_decay=0.995, eps_min=0.01, gamma=0.99, lr=0.1):
        self.num_actions = num_actions
        self.k = k
        self.max_mem = max_mem
        self.epsilon = epsilon
        self.eps_decay = eps_decay
        self.eps_min = eps_min
        self.gamma = gamma
        self.lr = lr

        self.cam = [[] for _ in range(num_actions)]
        self.ram = [[] for _ in range(num_actions)]
        self.ep_count = 0
        self.step_count = 0

    def store(self, state, action, q):
        self.cam[action].append(state.copy())
        self.ram[action].append(q)
        if len(self.cam[action]) > self.max_mem:
            self.cam[action].pop(0)
            self.ram[action].pop(0)

    def lookup(self, state, action):
        if len(self.cam[action]) == 0:
            return 0.0

        states = np.array(self.cam[action])
        dists = np.linalg.norm(states - state, axis=1)

        k = min(self.k, len(dists))
        idx = np.argpartition(dists, k-1)[:k]
        k_dists = dists[idx]
        k_qs = np.array([self.ram[action][i] for i in idx])

        weights = 1.0 / (k_dists + 1e-8)
        return np.average(k_qs, weights=weights)

    def predict(self, state, deterministic=False):
        if not deterministic and np.random.random() < self.epsilon:
            action = np.random.randint(self.num_actions)
            method = 'random'
        else:
            qs = np.array([self.lookup(state, a) for a in range(self.num_actions)])
            action = int(np.argmax(qs))
            method = 'greedy'
        return action, {'method': method, 'epsilon': self.epsilon}

    def update(self, state, action, reward, next_state, done):
        if done:
            target = reward
        else:
            next_qs = [self.lookup(next_state, a) for a in range(self.num_actions)]
            target = reward + self.gamma * max(next_qs)

        current = self.lookup(state, action)
        new_q = current + self.lr * (target - current)
        self.store(state, action, new_q)
        self.step_count += 1

    def decay_epsilon(self):
        self.epsilon = max(self.eps_min, self.epsilon * self.eps_decay)

    def get_statistics(self):
        sizes = [len(c) for c in self.cam]
        return {
            'episode_count': self.ep_count,
            'training_step': self.step_count,
            'epsilon': self.epsilon,
            'total_memory_size': sum(sizes),
            'memory_per_action': {f'action_{i}': s for i, s in enumerate(sizes)},
            'avg_q_values': {f'action_{i}': np.mean(r) if r else 0.0 for i, r in enumerate(self.ram)}
        }

    def save(self, path):
        data = {
            'num_actions': self.num_actions, 'k': self.k, 'max_mem': self.max_mem,
            'epsilon': self.epsilon, 'eps_decay': self.eps_decay, 'eps_min': self.eps_min,
            'gamma': self.gamma, 'lr': self.lr,
            'cam': self.cam, 'ram': self.ram,
            'ep_count': self.ep_count, 'step_count': self.step_count
        }
        with open(path, 'wb') as f:
            pickle.dump(data, f)
        print(f"saved to {path}")

    @staticmethod
    def load(path):
        with open(path, 'rb') as f:
            data = pickle.load(f)

        agent = CAMRAM(
            num_actions=data['num_actions'],
            k=data.get('k', data.get('k_neighbors', 5)),
            max_mem=data.get('max_mem', data.get('max_memory', 10000)),
            epsilon=data.get('epsilon', 0.1),
            eps_decay=data.get('eps_decay', data.get('epsilon_decay', 0.995)),
            eps_min=data.get('eps_min', data.get('epsilon_min', 0.01)),
            gamma=data.get('gamma', 0.99),
            lr=data.get('lr', data.get('learning_rate', 0.1))
        )
        agent.cam = data['cam']
        agent.ram = data['ram']
        agent.ep_count = data.get('ep_count', data.get('episode_count', 0))
        agent.step_count = data.get('step_count', data.get('training_step', 0))

        print(f"loaded from {path}")
        return agent

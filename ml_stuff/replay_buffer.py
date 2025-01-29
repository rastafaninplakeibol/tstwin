import numpy as np

class LSTMReplayBuffer:
    def __init__(self, max_size=10000):
        self.states = []
        self.actions = []
        self.next_states = []
        self.max_size = max_size
        self.ptr = 0

    def store(self, state, action, next_state):
        if len(self.states) < self.max_size:
            self.states.append(state)
            self.actions.append(action)
            self.next_states.append(next_state)
        else:
            idx = self.ptr % self.max_size
            self.states[idx] = state
            self.actions[idx] = action
            self.next_states[idx] = next_state
        self.ptr += 1

    def sample_batch(self, batch_size):
        if len(self.states) < batch_size:
            return None, None, None
        idxs = np.random.randint(0, len(self.states), size=batch_size)
        s = np.array([self.states[i] for i in idxs])
        a = np.array([self.actions[i] for i in idxs])
        ns = np.array([self.next_states[i] for i in idxs])
        return s, a, ns

    def size(self):
        return len(self.states)

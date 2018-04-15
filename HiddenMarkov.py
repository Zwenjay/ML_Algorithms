import numpy as np

class HiddenMarkovModel(object):
    # Calculation About Hidden Markov Model.
    #     Including following functions:
    #         - calculate_prob
    #             given the observations, calculate the probability of the observations.
    #         - get_sequence
    #             given the observations, return the most likely hidden states using Viterbi algorithm.
    #         - predict_next
    #             given the observations, predict which is most likely observation in the next turn

    def __init__(self, A, p, pi, O):
    #     Attributes:
    #         A: 
    #             The transformation matrix of hidden states.
    #             If the number of hidden state is K, it should be a K * K ndarray
    #             For example:
    #                 np.array([[0.2, 0.3], [0.8, 0.7]])
    #         p: 
    #             The probabilities of observations given a certain hidden state.
    #             If the number of hidden state is K and the number of observations is N, it should be a K * N ndarray
    #             For example:
    #             np.array([[0.2, 0.3, 0.5], [0.8, 0.1, 0.1])
    #         pi:
    #             The initial probabilities of each hidden states.
    #             If the number of hidden state is K, it should be an 1 * K ndarray.
    #             For example:
    #             np.array([0.6, 0.4])
    #         O:
    #             The sequence of observations.
    #             If the length of sequence is T, it should be an 1 * T 1 based integer ndarray.
    #             For example:
    #                 np.array([1, 2, 3, 2, 1])
    # 
        self.A = A
        self.p = p
        self.pi = pi
        self.O = O
        self.last_prob = None

    def calculate_prob(self):
        """Calculate the probability of a given sequence of observations.
            Returns:
                A float, the probability of observing such a sequence.
        """
        num_z = self.A.shape[0]
        p = self.pi
        res = 1
        for x in self.O:
            prob = p * self.p[:, x-1]
            prob = sum(prob)
            res *= prob
            p = np.dot(p , self.A)
            total = np.sum(p)
            p = p / total
            self.last_prob = prob
        return res 


    def get_sequence(self):
    # Return the most likely hidden states using Viterbi algorithm.
    #     Returns:
    #         An 1-D ndarray of 0-based integer, the sequence of the most likely hidden states
    #         For example:
    #             np.array([1, 0, 2, 0, 1])


        num_z = self.A.shape[0]
        delta = {}
        delta[0] = self.pi
        last = {}
        for i, x in enumerate(self.O):
            delta[i+1] = np.zeros(num_z)
            last[i+1] = np.zeros(num_z)
            for k in range(num_z):
                temp = delta[i] * self.A[:, k] * self.p[k][x-1]
                delta[i+1][k] = np.max(temp, axis=0)
                last[i+1][k] = np.argmax(temp, axis=0)

        state = []
        s = np.argmax(delta[len(X)])
        state.append(s)
        for i in reversed(range(1, len(X+1))):
            state.append(int(last[i][state[-1]]))
        state.reverse()
        return state

    def predict_next(self):
        """Return the most likely hidden states using Viterbi algorithm.
        Returns:
            An integer denote the next most likely observation
        """
        if not self.last_prob is None:
            _ = self.calculate_prob()
        return np.argmax(self.last_prob)



# ================Test=========================


A = np.array([[0.2, 0.8],[0.4, 0.6]])
t = np.array([[0.4, 0.1],[0.2, 0.3]])
pi = np.array([0.7, 0.3])
X = np.array([1,1,2,1,2,1])
p = 1

hmm = HiddenMarkovModel(A, t, pi, X)
s = hmm.get_sequence()
p = hmm.calculate_prob()
next_ob = hmm.predict_next()
print(s)
print(p)
print(next_ob)



import numpy as np


class HiddenMarkovModel(object):
	"""Calculation About Hidden Markov Model.
	Including following functions:
		- calculateProb
			given the parameters, and observations, calculate the probability of the observations.
		- getSequence
			given the parameters and the observations, return the most likely hidden states using Viterbi algorithm.
	"""
	def __init__(self, A, p, pi):
		""" Init HiddenMarkovModel with A, p, pi.
		Attributes:
	        A: 
	        	The transformation matrix of hidden states.
	        	If the number of hidden state is K, it should be a K * K ndarray
	        	For example:
	        		np.array([[0.2, 0.3], [0.8, 0.7]])
	        p: 
	        	The probabilities of observations given a certain hidden state.
	        	If the number of hidden state is K and the number of observations is N, it should be a K * N ndarray
	        	For example:
	        		np.array([[0.2, 0.3, 0.5], [0.8, 0.1, 0.1])
	        pi:
	        	The initial probabilities of each hidden states.
	        	If the number of hidden state is K, it should be an 1 * K ndarray.
	        	For example:
	        	np.array([0.6, 0.4])
	    """
		self.A = A
		self.p = p
		self.pi = pi

	def calculateProb(self, O):
		"""Calculate the probability of a given sequence of observations.
	    Args:
	    	O: The sequence of observations, it should be an 1-D ndarray, observations are 1 based integers.
	    	For example:
	    		np.array([2, 1, 3, 1, 2])
	    Returns:
	    	A float, the probability of observing such a sequence.
	    """

		num_z = self.A.shape[0]
		p = self.pi
		res = 1
		for x in O:		
			prob = sum(p * self.p[:, x-1])
			res *= prob
			p = np.dot(p , self.A)
			total = np.sum(p)
			p = p / total
		return res 


	def getSequence(self, O):
		"""Return the most likely hidden states using Viterbi algorithm.
	    Args:
	    	O: The sequence of observations, it should be an 1-D ndarray, observations are 1 based integers.
	    	For example:
	    		np.array([2, 1, 3, 1, 2])
	    Returns:
	    	An 1-D ndarray of 0-based integer, the sequence of the most likely hidden states
	    	For example:
	    		np.array([1, 0, 2, 0, 1])
    """


		num_z = self.A.shape[0]
		delta = {}
		delta[0] = self.pi
		last = {}
		for i, x in enumerate(O):
			delta[i+1] = np.zeros(num_z)
			last[i+1] = np.zeros(num_z)
			for k in range(num_z):
				temp = delta[i] * self.A[:, k] * self.p[k][x-1]
				delta[i+1][k] = np.max(temp, axis=0)
				last[i+1][k] = np.argmax(temp, axis=0)
				# for j in range(num_z):
				# 	if delta[i][j] * self.A[j][k] * self.p[k][x-1] > delta[i+1][k]:
				# 		delta[i+1][k] = delta[i][j] * A[j][k] * self.p[k][x-1]
				# 		last[i+1][k] = j

		state = []
		s = np.argmax(delta[len(X)])
		state.append(s)
		for i in reversed(range(1, len(X+1))):
			state.append(int(last[i][state[-1]]))
		state.reverse()
		return state


# ================Test=========================


A = np.array([[0.2, 0.8],[0.4, 0.6]])
t = np.array([[0.4, 0.1],[0.2, 0.3]])
pi = np.array([0.7, 0.3])
X = np.array([1,1,2,1,2,1])
p = 1

hmm = HiddenMarkovModel(A, t, pi)
s = hmm.getSequence(X)
p = hmm.calculateProb(X)
print(s)
print(p)


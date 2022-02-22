# fourier_fa.py - Python implementation of the Fourier Basis
#
# Source paper (please cite!): 
#  G.D. Konidaris, S. Osentoski and P.S. Thomas. Value Function 
#  Approximation in Reinforcement Learning using the Fourier Basis. 
#  In Proceedings of the Twenty-Fifth AAAI Conference on Artificial 
#  Intelligence, pages 380-385, August 2011.
#
# Note: this class computes the coefficient matrix at instantiation,
# and stores a matrix of size c \times d (where c is the number of 
# basis functions and d is the input state dimensionality) in memory 
# for the lifetime of the class instance. That can cause memory 
# issues since c grows exponentially with d and the basis order. An 
# alternative implementation (not given here) is to recompute the 
# coefficients whenever the basis must be evaluated. This only 
# ever stores O(c) data in memory, at the cost of more expensive 
# evaluation computation.  
#
# Author: George Konidaris (gdk@cs.brown.edu)
# Creation Date: February 22nd 2021 
# Copyright 2021 Brown University
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
#

import numpy as np
import sys
import itertools
from timeit import default_timer as timer

class FourierBasis:
    """
    A class implementing the Fourier Basis.

    Methods
    -------
    length()
        returns the number of basis functions
    get_coefficients()
        returns the Fourier Basis coefficients
    get_coefficient(function_no)
        returns the coefficients of a specific Fourier Basis function 
    get_gradient_factors()
        return the gradient scaling factors
    get_gradient_factor(function_no)
        return a specific gradient scaling factor
    evaluate(state_vector)
        evaluate the basis functions at a given state

    Attributes (should not typically be accessed directly)
    ----------
    d : integer
        the dimensionality of the input state
    order : numpy array of d elements
        the Fourier basis order for each state dimension
    independent: boolean
        indicates whether state variables are considered separately 
        (True) or cross-variable dependencies are included (False)
    coefficients : 2D numpy array 
        Fourier basis coefficients; k x d array, where k is the 
        number of basis functions and d the state dimensionality
    gradient_factors: 2D numpy array
        Fourier basis gradient scaling; same dimensionality as 
        coefficients
    """

    def __init__(self, order, d = None, independent=False):
        """
        Constructs a Fourier Basis instantiation. 

        Note that this implementation constructs a Numpy array of all 
        the Fourier coefficients, which supports efficient basis 
        computation but is memory-heavy - O(bd), where b is the number
        of basis functions and d is the state dimensionality. An 
        alternative implementation recomputes each coefficients vector
        at evaluation time, and is only O(d) memory. 

        Parameters
        ----------
        order : int or list of d ints
            either the maximum coefficient attached to any state 
            variable, or a list of such maximial values for each state
            variable. If the latter, d can be omitted.
        d : int, optional
            the dimensionality of the state space; if None, obtained by
            len(order) (order must in that case be a list).
        independent: boolean, optional
            whether or not the dimensions are tiled independently; 
            False by default. If True, the size of basis is not 
            exponential in d and order. 
        """ 

        # Instance variables
        self.coefficients = np.array([])
        self.gradient_factors = np.array([])
        self.order = []
        self.d = 0
        self.independent = independent

        # Input type checking
        if(d is None):
            # If no dimensionality specified, get it from order parameter (must be able to call len).
            if(type(order) in [list,tuple,np.ndarray]):
                self.d = len(order)
                self.order = order
            else:
                print("FourierBasis error: must either specify input dimension 'd' or order must provide a d-length list of coefficient order in `order'.", file = sys.stderr)
                return
        else:
            # If dimensionality specified, confirm against 'order'.
            # If 'order' is an integer not a list, assume that's the order in each of 'd' dimensions.
            self.d = d
            if(type(order) in [list,tuple,np.ndarray]):
                if(d != len(order)):
                    print("FourierBasis error: parameter 'd' does not match length of coefficient order list `order'.", file = sys.stderr)
                    return
                self.order = order
            elif(type(order) != int):
                print("FourierBasis error: parameter 'order' must be an integer, if not a list.", file = sys.stderr)
                return
            else:
                # Order list should be 'd' copies of the integer in 'order'
                self.order = [order]*d
                
        # Generate coefficients
        if(self.independent):
            coeffs = []
            coeffs.append(np.zeros([self.d]))
            for i in range(0, self.d):
                for c in range(0, self.order[i]):
                    v = np.zeros([self.d])
                    v[i] = c + 1
                    coeffs.append(v)
        else:
            coeffs = []
            prods = [range(0, o+1) for o in self.order]
            coeffs = [v for v in itertools.product(*prods)]

        self.coefficients = np.array(coeffs)
        self.gradient_factors = 1.0 / np.linalg.norm(self.coefficients, ord=2, axis=1)
        self.gradient_factors[0] = 1.0 # Overwrite division by zero for function with all-zero coefficients.

    def length(self):
        """Return the number of basis functions.""" 

        return self.coefficients.shape[0]

    def get_coefficients(self):
        """Returns an b x d numpy array of coefficients, where b
        is the number of basis functions, and is the state dimension."""

        return self.coefficients

    def get_coefficient(self, function_no):
        """ Returns the coefficients (a d-dimensional numpy array,
        where d is the state dimensionality) for the individual
        basis function specified by function_no."""

        return self.coefficients[function_no, :]

    def get_gradient_factors(self):
        """ Returns a numpy array of the gradient correction terms for
        all basis functions."""

        return self.gradient_factors

    def get_gradient_factor(self, function_no):
        """ Returns the gradient correction term for the specific basis
        function function_no."""

        return self.gradient_factors[function_no]
        
    def evaluate(self, state_vector):
        """
        Computes basis function values at a given state. 

        Parameters
        ----------
        state_vector : a list of d real-values
            d is state dimensionality; each value should be in [0, 1]
            If the list is not a numpy array, it is converted to one.

        Returns
        -------
        numpy array
            An n-dimensional numpy array containing a basis function
            value for each of the n functions in the basis. 
        """ 

        # Type check state vector, try to convert if not ndarray.
        if(type(state_vector) is not np.ndarray):
            state_vector = np.array(state_vector)
            
        # Dimension check state vector
        state_shape = state_vector.shape
        if(not ((state_shape == (self.d, )) or (state_shape == (self.d, 1)))):
            print("FourierBasis warning: state vector shape is " + str(state_shape) + ", expecting " + str((self.d, )) + " or " + str((self.d, 1)), file = sys.stderr)
            return None
        
        # Bounds check state vector
        if(np.min(state_vector) < 0.0):
            print("FourierBasis warning: given state vector with entry less than 0 (" + str(np.min(state_vector)) + " at index " + str(np.argmin(state_vector)) + ").", file = sys.stderr)
        if(np.max(state_vector) > 1.0):
            print("FourierBasis warning: given state vector with entry greater than 1 (" + str(np.max(state_vector)) + " at index " + str(np.argmax(state_vector)) + ").", file = sys.stderr)

        # Compute the Fourier Basis feature values 
        return np.cos(np.pi*np.dot(self.coefficients, state_vector))

if __name__ == "__main__":

    start = timer()
    fb = FourierBasis(order=7, d=7)
    end = timer()
    print("Constructed", fb.length(), "basis functions in", end-start, " seconds")

    start = timer()
    fb.evaluate([0.1]*7)
    end = timer()
    print(end - start, "seconds to evaluate", fb.length(), "basis functions")

    start = timer()
    fb.evaluate([0.3]*7)
    end = timer()
    print(end - start, "seconds to evaluate", fb.length(), "basis functions")

    start = timer()
    fb.evaluate([0.25]*7)
    end = timer()
    print(end - start, "seconds to evaluate", fb.length(), "basis functions")

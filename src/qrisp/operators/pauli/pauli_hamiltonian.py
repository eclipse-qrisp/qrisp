"""
\********************************************************************************
* Copyright (c) 2024 the Qrisp authors
*
* This program and the accompanying materials are made available under the
* terms of the Eclipse Public License 2.0 which is available at
* http://www.eclipse.org/legal/epl-2.0.
*
* This Source Code may also be made available under the following Secondary
* Licenses when the conditions for such availability set forth in the Eclipse
* Public License, v. 2.0 are satisfied: GNU General Public License, version 2
* with the GNU Classpath Exception which is
* available at https://www.gnu.org/software/classpath/license.html.
*
* SPDX-License-Identifier: EPL-2.0 OR GPL-2.0 WITH Classpath-exception-2.0
********************************************************************************/
"""
from qrisp.operators.hamiltonian import Hamiltonian
#from qrisp.operators.pauli.helper_functions import *
#from qrisp.operators.pauli.pauli_term import PauliTerm
#from qrisp.operators.pauli.pauli_measurement import PauliMeasurement
from abc import ABC, abstractmethod
import sympy as sp

from sympy import init_printing
# Initialize automatic LaTeX rendering
init_printing()

threshold = 1e-9

#
# PauliHamiltonian
#

class PauliHamiltonian(Hamiltonian):
    r"""
    This class provides an efficient implementation of Pauli Hamiltonians, i.e.,
    Hamiltonians of the form

    .. math::
        
        H=\sum\limits_{j}\alpha_jP_j 
            
    where $P_j=\prod_i\sigma_i^j$ is a Pauli product, 
    and $\sigma_i^j\in\{I,X,Y,Z\}$ is the Pauli operator acting on qubit $i$.

    Examples
    --------

    A ``UnboundPauliHamiltonian`` can be specified conveniently in terms of ``X``, ``Y``, ``Z`` operators:

    ::
        
        from qrisp.operators.pauli import X,Y,Z

        H = 1+2*X(0)+3*X(0)*Y(1)
        H

    Yields $1+2X_0+3X_0Y_1$.


    A ``BoundPauliHamiltonian`` can be specified conveniently in terms of ``X``, ``Y``, ``Z`` operators:

    ::

        from qrisp import QuantumVariable
        from qrisp.operators import BoundPauliHamiltonian, X,Y,Z
        
        qv = QuantumVariable(2)
        H = 1+2*X(qv[0])+3*X(qv[0])*Y(qv[1])

    Yields $1+2X(qv.0)+3X(qv.0)Y(qv.1)$.

    """
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def len(self):
        pass
    
    #
    # Printing
    #

    @abstractmethod
    def _repr_latex_(self):
        pass
    
    @abstractmethod
    def __str__(self):
        pass

    @abstractmethod
    def to_expr(self):
        """
        Returns a SymPy expression representing the operator.

        Returns
        -------
        expr : sympy.expr
            A SymPy expression representing the operator.

        """
        pass

    #
    # Arithmetic
    #

    @abstractmethod
    def __add__(self,other):
        """
        Returns the sum of the operator self and other.

        Parameters
        ----------
        other : int, float, complex or PauliHamiltonian
            A scalar or a PauliHamiltonian to add to the operator self.

        Returns
        -------
        result : PauliHamiltonian
            The sum of the operator self and other.

        """
        pass

    @abstractmethod
    def __sub__(self,other):
        """
        Returns the difference of the operator self and other.

        Parameters
        ----------
        other : int, float, complex or PauliHamiltonian
            A scalar or a PauliHamiltonian to substract from the operator self.

        Returns
        -------
        result : PauliHamiltonian
            The difference of the operator self and other.

        """
        pass
    
    @abstractmethod 
    def __rsub__(self,other):
        """
        Returns the difference of the operator other and self.

        Parameters
        ----------
        other : int, float, complex or PauliHamiltonian
            A scalar or a PauliHamiltonian to substract the operator self from.

        Returns
        -------
        result : PauliHamiltonian
            The difference of the operator other and self.

        """
        pass

    @abstractmethod 
    def __mul__(self,other):
        """
        Returns the product of the operator self and other.

        Parameters
        ----------
        other : int, float, complex or PauliHamiltonian
            A scalar or a PauliHamiltonian to multiply with the operator self.

        Returns
        -------
        result : PauliHamiltonian
            The product of the operator self and other.

        """
        pass

    __radd__ = __add__
    __rmul__ = __mul__

    #
    # Inplace arithmetic
    #
    @abstractmethod 
    def __iadd__(self,other):
        """
        Adds other to the operator self.

        Parameters
        ----------
        other : int, float, complex or PauliHamiltonian
            A scalar or a PauliHamiltonian to add to the operator self.

        """
        pass       

    @abstractmethod 
    def __isub__(self,other):
        """
        Substracts other from the operator self.

        Parameters
        ----------
        other : int, float, complex or PauliHamiltonian
            A scalar or a PauliHamiltonian to substract from the operator self.

        """
        pass

    @abstractmethod  
    def __imul__(self,other):
        """
        Multiplys other to the operator self.

        Parameters
        ----------
        other : int, float, complex or PauliHamiltonian
            A scalar or a PauliHamiltonian to multiply with the operator self.

        """
        pass

    #
    # Substitution
    #
    @abstractmethod 
    def subs(self, subs_dict):
        """
        
        Parameters
        ----------
        subs_dict : dict
            A dictionary with indices (int) as keys and numbers (int, float, complex) as values.

        Returns
        -------
        result : PauliHamiltonian
            The resulting PauliHamiltonian.
        
        """
        pass

    #
    # Miscellaneous
    #
    @abstractmethod 
    def apply_threshold(self,threshold):
        """
        Removes all Pauli terms with coefficient absolute value below the specified threshold.

        Parameters
        ----------
        threshold : float
            The threshold for the coefficients of the Pauli terms.

        """
        pass

    @abstractmethod 
    def to_sparse_matrix(self):
        """
        Returns a matrix representing the operator.
    
        Returns
        -------
        M : scipy.sparse.csr_matrix
            A sparse matrix representing the operator.

        """
        pass

    @abstractmethod 
    def ground_state_energy(self):
        """
        Calculates the ground state energy (i.e., the minimum eigenvalue) of the operator classically.
    
        Returns
        -------
        E : float
            The ground state energy. 

        """
        pass
    
    #
    # Partitions 
    #

    # Commutativity: Partitions the PauliHamiltonian into PauliHamiltonians with pairwise commuting PauliTerms
    @abstractmethod 
    def commuting_groups(self):
        r"""
        Partitions the PauliHamiltonian into PauliHamiltonians with pairwise commuting terms. That is,

        .. math::

            H = \sum_{i=1}^mH_i

        where the terms in each $H_i$ are mutually commuting.

        Returns
        -------
        groups : list[PauliHamiltonian]
            The partition of the Hamiltonian.
        
        """
        pass

    # Qubit-wise commutativity: Partitions the PauliHamiltonian into PauliHamiltonians with pairwise qubit-wise commuting PauliTerms
    @abstractmethod 
    def commuting_qw_groups(self, show_bases=False):
        r"""
        Partitions the PauliHamiltonian into PauliHamiltonians with pairwise qubit-wise commuting terms. That is,

        .. math::

            H = \sum_{i=1}^mH_i

        where the terms in each $H_i$ are mutually qubit-wise commuting.

        Returns
        -------
        groups : list[PauliHamiltonian]
            The partition of the Hamiltonian.
        
        """
        pass
    
    #
    # Measurement settings
    #
    @abstractmethod 
    def pauli_measurement(self):
        pass

    #
    # Trotterization
    #
    @abstractmethod 
    def trotterization(self):
        r"""
        Returns a function for performing Hamiltonian simulation, i.e., approximately implementing the unitary operator $e^{itH}$ via Trotterization.

        Returns
        -------
        U : function 
            A Python function that implements the first order Suzuki-Trotter formula.
            Given a Hamiltonian $H=H_1+\dotsb +H_m$ the unitary evolution $e^{itH}$ is 
            approximated by 
            
            .. math::

                e^{itH}\approx U_1(t,N)=\left(e^{iH_1t/N}\dotsb e^{iH_mt/N}\right)^N

            This function receives the following arguments:

            * qarg : QuantumVariable 
                The quantum argument.
            * t : float, optional
                The evolution time $t$. The default is 1.
            * steps : int, optional
                The number of Trotter steps $N$. The default is 1.
            * iter : int, optional 
                The number of iterations the unitary $U_1(t,N)$ is applied. The default is 1.
        
        """
        pass
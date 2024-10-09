"""
\********************************************************************************
* Copyright (c) 2023 the Qrisp authors
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

import jax.numpy as jnp
class JRangeIterator:
    
    def __init__(self, stop):
        
        self.stop = stop
        
    def __iter__(self):
        self.iteration = 0
        return self
    
    def __next__(self):
        
        self.iteration += 1
        if self.iteration == 1:
            from qrisp.environments import JIterationEnvironment
            self.iter_env = JIterationEnvironment()
            self.iter_env.__enter__()
            self.stop += 1
            return self.stop
        elif self.iteration == 2:
            self.iter_env.__exit__(None, None, None)
            self.iter_env.__enter__()
            self.stop += 1
            return self.stop
        elif self.iteration == 3:
            self.iter_env.__exit__(None, None, None)
            raise StopIteration

def jrange(stop):
    """
    .. _jrange:
    
    Performs a loop with a dynamic bound.
    
    .. warning::
        
        Similar to the :ref:`ClControlEnvironment <ClControlEnvironment>`, this feature must not have
        external carry values, implying values computed within the loop can't 
        be used outside of the loop. It is however possible to carry on values 
        from the previous iteration.
    
    .. warning::
        
        Each loop iteration must perform exactly the same instructions - the only
        thing that changes is the loop index
    

    Parameters
    ----------
    stop : int
        The loop index to stop at.

    Examples
    --------
    
    We construct a function that encodes an integer into an arbitrarily sized
    :ref:`QuantumVariable`:
        
    ::
        
        from qrisp import QuantumFloat, control, x
        from qrisp.jasp import jrange, make_jaspr
        
        @qache
        def int_encoder(qv, encoding_int):
            
            for i in jrange(qv.size):
                with control(encoding_int & (1<<i)):
                    x(qv[i])

        def test_f(a, b):
            
            qv = QuantumFloat(a)
            
            int_encoder(qv, b+1)
            
            return measure(qv)
            
        jaspr = make_jaspr(test_f)(1,1)
    
    Test the result:
        
    >>> jaspr(5, 8)
    9
    >>> jaspr(5, 9)
    10
    
    We now give examples that violate the above rules (ie. no carries and changing
    iteration behavior).
    
    To create a loop with carry behavior we simply return the final loop index
    
    ::
        
        @qache
        def int_encoder(qv, encoding_int):
            
            for i in jrange(qv.size):
                with control(encoding_int & (1<<i)):
                    x(qv[i])
            return i
            

        def test_f(a, b):
            
            qv = QuantumFloat(a)
            
            int_encoder(qv, b+1)
            
            return measure(qv)
            
        jaspr = make_jaspr(test_f)(1,1)

    >>> jaspr(5, 8)
    Exception: Found jrange with external carry value
    
    To demonstrate the second kind of illegal behavior, we construct a loop
    that behaves differently on the first iteration:
        
    ::
        
        @qache
        def int_encoder(qv, encoding_int):
            
            flag = True
            for i in jrange(qv.size):
                if flag:
                    with control(encoding_int & (1<<i)):
                        x(qv[i])
                else:
                    x(qv[0])
                flag = False
            
        def test_f(a, b):
            
            qv = QuantumFloat(a)
            
            int_encoder(qv, b+1)
            
            return measure(qv)
            
        jaspr = make_jaspr(test_f)(1,1)

    In this script, ``int_encoder`` defines a boolean flag that changes the 
    semantics of the iteration behavior. After the first iteration the flag
    is set to ``False`` such that the alternate behavior is activated.
    
    >>> jaspr(5, 8)
    Exception: Jax semantics changed during jrange iteration

    """
    if isinstance(stop, int):
        return range(stop)
    else:
        return JRangeIterator(stop)
    
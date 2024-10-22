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

from qrisp.circuit.qubit import Qubit
from qrisp.operators.pauli.unbound_pauli_hamiltonian import UnboundPauliHamiltonian
from qrisp.operators.pauli.bound_pauli_hamiltonian import BoundPauliHamiltonian
from qrisp.operators.pauli.unbound_pauli_term import UnboundPauliTerm
from qrisp.operators.pauli.bound_pauli_term import BoundPauliTerm

def X(arg):
    if isinstance(arg,int):
        return UnboundPauliHamiltonian({UnboundPauliTerm({arg:"X"}):1})
    elif isinstance(arg, Qubit):
        return BoundPauliHamiltonian({BoundPauliTerm({arg:"X"}):1})
    else:
        raise Exception("Cannot initialize operator from type "+str(type(arg)))

def Y(arg):
    if isinstance(arg,int):
        return UnboundPauliHamiltonian({UnboundPauliTerm({arg:"Y"}):1})
    elif isinstance(arg, Qubit):
        return BoundPauliHamiltonian({BoundPauliTerm({arg:"Y"}):1})
    else:
        raise Exception("Cannot initialize operator from type "+str(type(arg)))

def Z(arg):
    if isinstance(arg,int):
        return UnboundPauliHamiltonian({UnboundPauliTerm({arg:"Z"}):1})
    elif isinstance(arg, Qubit):
        return BoundPauliHamiltonian({BoundPauliTerm({arg:"Z"}):1})
    else:
        raise Exception("Cannot initialize operator from type "+str(type(arg)))
    








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

from qrisp import *
from qrisp.jasp import *

def test_dynamic_mcx():
    
    @terminal_sampling
    def main(i, j):
        qf = QuantumFloat(i)
        h(qf)
        qbl = QuantumBool()
        mcx(qf.reg, qbl[0], method = "balauca", ctrl_state = j)
        return qf, qbl
    
    for i in range(1, 5):
        for j in range(2**i):
            res_dict = main(i, j)
            
            for k in res_dict.keys():
                if k[0] == j:
                    assert k[1]
                else:
                    assert not k[1]
        
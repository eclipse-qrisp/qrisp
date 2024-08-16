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

from jax import jit, make_jaxpr
from jax.core import Jaxpr, JaxprEqn, ClosedJaxpr, Var
from qrisp.jax import get_tracing_qs, check_for_tracing_mode, flatten_collected_environments, eval_jaxpr, AbstractQuantumCircuit

def control_eqn(eqn, ctrl_qubit_var):
    """
    Receives and equation that describes either an operation or a pjit primitive
    and returns an equation that describes the inverse.

    Parameters
    ----------
    eqn : jax.core.JaxprEqn
        The equation to be inverted.

    Returns
    -------
    inverted_eqn
        The equation with inverted operation.

    """
    if eqn.primitive.name == "pjit":
        eqn.params["jaxpr"] = ClosedJaxpr(control_jispr(eqn.params["jaxpr"].jaxpr, ctrl_qubit_var),
                                          eqn.params["jaxpr"].consts)
        return eqn
    else:
        return JaxprEqn(primitive = eqn.primitive.control(),
                        invars = [eqn.invars[0], ctrl_qubit_var] + eqn.invars[1:],
                        outvars = eqn.outvars,
                        params = eqn.params,
                        source_info = eqn.source_info,
                        effects = eqn.effects,
                        ctx = eqn.ctx)

def control_jispr(jispr):
    """
    Takes a jaxpr returning a quantum circuit and returns a jaxpr, which returns
    the inverse quantum circuit

    Parameters
    ----------
    jaxpr : jax.core.Jaxpr
        A jaxpr returning a QuantumCircuit.

    Returns
    -------
    inverted_jaxpr : jaxpr.core.Jaxpr
        A jaxpr returning the inverse QuantumCircuit.

    """
    
    from qrisp.circuit import Operation
    from qrisp.jax import Jispr, AbstractQubit
    
    ctrl_qubit_var = Var(suffix = "", aval = AbstractQubit())
    
    new_eqns = []
    for eqn in jispr.eqns:
        if isinstance(eqn.primitive, Operation) or eqn.primitive.name == "pjit":
            new_eqns.append(control_eqn(eqn, ctrl_qubit_var))
        elif eqn.primitive.name == "measure":
            raise Exception("Tried to applied quantum control to a measurement")
        else:
            new_eqns.append(eqn)
    
    
    permeability = dict(jispr.permeability)
    permeability[ctrl_qubit_var] = True
    
    return Jispr(permeability = permeability,
                 isqfree = jispr.isqfree,
                 constvars = [ctrl_qubit_var] + jispr.constvars, 
                 invars = jispr.invars, 
                 outvars = jispr.outvars, 
                 eqns = new_eqns)
        
def multi_control_jispr(jispr, num_ctrl = 1, ctrl_state = -1):
    
    if num_ctrl == 1:
        return control_jispr(jispr)
    
    from qrisp.jax import Jispr, AbstractQubit
    
    ctrl_vars = [Var(suffix = "", aval = AbstractQubit()) for _ in range(num_ctrl)]
    ctrl_avals = [x.aval for x in ctrl_vars]
    
    temp_jaxpr = make_jaxpr(exec_multi_controlled_jispr(jispr))(ctrl_avals, *[var.aval for var in jispr.invars + jispr.constvars]).jaxpr
    
    invars = temp_jaxpr.invars[num_ctrl:-len(jispr.constvars)]
    constvars = temp_jaxpr.invars[:num_ctrl] + temp_jaxpr.invars[-len(jispr.constvars):]
    
    res = Jispr(
                 invars = invars,
                 constvars = constvars,
                 outvars = temp_jaxpr.outvars,
                 eqns = temp_jaxpr.eqns)
    
    return res
    
    
def exec_multi_controlled_jispr(jispr):
    
    def multi_controlled_jispr_executor(ctrls, *args):
        
        if len(ctrls) == 1:
            controlled_jispr = control_jispr(jispr)
            return eval_jaxpr(controlled_jispr)(ctrls[0], *args)
            
        else:
            from qrisp.circuit import XGate
            from qrisp import QuantumBool
            
            qs = get_tracing_qs()
            args = list(args)
            qs.abs_qc = args.pop(0)
            
            invalues = []
            for i in range(len(jispr.invars)-1):
                invalues.append(args.pop(0))
            constvalues = args
            
            controlled_jispr = control_jispr(jispr)
            mcx_operation = XGate().control(len(ctrls))
            
            ctrl_qbl = QuantumBool()
            ctrl_qb = ctrl_qbl[0]
            
            qs.append(mcx_operation, ctrls + [ctrl_qb])
            
            res = controlled_jispr.eval(*(invalues + [ctrl_qb] + constvalues))
            
            qs.append(mcx_operation, ctrls + [ctrl_qb])
            
            return qs.abs_qc, res
            
    return multi_controlled_jispr_executor
        
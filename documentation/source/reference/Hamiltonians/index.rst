Hamiltonians
============

.. note::

    This module is still under heavy development and the interface can therefore change at any time!

This Hamiltonians submodule of Qrisp provides a unified framework to describe, optimize and simulate quantum Hamiltonians.
It provides a collection of different types of Hamiltonians that can be used to model and solve a variety of problems in physics, chemistry, or optimization.
(Up to now, Pauli Hamiltonians have been implemented, but stay tuned for future updates!)
Each type of Hamiltonian comes with comprehensive documentation and brief examples to help you understand its implementation and usage:

.. list-table::
   :widths: 40 60
   :header-rows: 1

   * - Hamiltonian
     - USED FOR
   * - :ref:`PauliHamiltonian <PauliHamiltonian>`
     - describe Hamiltonians in terms of Pauli operators

We encourage you to explore these Hamiltonians, delve into their documentation, and experiment with their implementations.

.. currentmodule:: qrisp

.. toctree::
   :hidden:
   
   PauliHamiltonian
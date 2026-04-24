import numpy as np
import pytest

from centrex_tlf import couplings, states


class TestCouplingMatrixEdgeCases:
    def test_qn_not_list_raises_type_error(self):
        from centrex_tlf.couplings.coupling_matrix import _generate_coupling_matrix_python

        ground = [1.0 * s for s in states.generate_coupled_states_ground(Js=[0])]
        excited = [1.0 * s for s in states.generate_coupled_states_excited(Js=[1], Ps=-1, Omegas=1)]
        QN = ground + excited
        with pytest.raises(TypeError, match="list"):
            _generate_coupling_matrix_python(
                ground_states=ground,
                excited_states=excited,
                QN=tuple(QN),
                pol_vec=np.array([0, 0, 1], dtype=complex),
            )

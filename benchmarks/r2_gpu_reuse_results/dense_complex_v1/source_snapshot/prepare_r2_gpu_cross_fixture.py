"""Prepare the frozen CPU benchmark recipe, without running CPU ODEs."""
import os
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[key]='1'
import json
from pathlib import Path
import numpy as np
from scipy.sparse import csr_matrix
from cross_system_fixtures import build_fixture
from centrex_tlf import states
from centrex_tlf.centrex_tlf_rust import create_lindblad_rhs_evaluator_py

OUT=Path(__file__).resolve().parent/'r2_gpu_reuse_results/matched_cross_v1'
if __name__=='__main__':
    OUT.mkdir(parents=True,exist_ok=True)
    f=build_fixture('r2_f4');q=f.prepared;n=q.layout.n
    ev=create_lindblad_rhs_evaluator_py(q.rust_plan,'expanded_sparse',True)
    rows,cols,values=ev.jacobian_packed_sparse_py(0.,0.,'analytic')
    for x,y in zip((rows,cols,values),ev.jacobian_packed_sparse_py(.71e-6,0.,'analytic')):np.testing.assert_array_equal(x,y)
    L=csr_matrix((values,(rows,cols)),shape=(n*n,n*n)).toarray()
    ground=np.asarray(states.QuantumSelector(electronic=states.ElectronicState.X,J=2).get_indices(f.system.QN),int).ravel()
    assert len(ground)==20
    packed=np.zeros((n*n,len(ground)));packed[ground,np.arange(len(ground))]=1.
    mask=np.zeros(n)
    det_slot=q.parameter_graph['slot_names'].index('detuning')
    det_slots={det_slot}
    for compound in q.parameter_graph['compounds']:
        ins=compound['expression']['instructions']
        if len(ins)==1 and ins[0]['op']==2 and ins[0].get('index') in det_slots:det_slots.add(compound['slot'])
    for coefficient in q.hamiltonian_plan['coefficients']:
        instructions=coefficient['expression']['instructions']
        if len(instructions)==1 and instructions[0]['op']==2 and instructions[0].get('index') in det_slots:
            for term in coefficient['basis_terms']:
                assert term['i']==term['j'] and term['im']==0.
                mask[term['i']]+=term['re']
    assert np.any(mask)
    np.savez(OUT/'inputs.npz',L=L,n=n,packed=packed,excited=np.array(f.excited_indices),times=np.array([0.,1e-6]),detuning_mask=mask)
    (OUT/'metadata.json').write_text(json.dumps({'configuration':f.configuration,'metadata':f.metadata,'initial_index':f.initial_index,'ground_indices':ground.tolist()},indent=2),encoding='utf-8')
    archive=Path(__file__).resolve().parent/'rc_propagation_results/cross_system_screen_v1/nominal/r2_f4/reference.npz'
    print('Prepared',n,'states; archived reference keys:',[(k,v.shape) for k,v in dict(np.load(archive)).items()],flush=True)

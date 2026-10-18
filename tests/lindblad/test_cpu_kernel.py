"""CPU kernels retain coherences, tails, overrides and native output semantics."""
from __future__ import annotations

import copy

import numpy as np
import pytest
import sympy as sp

from centrex_tlf.lindblad import grid_scan, solve_lindblad, solve_lindblad_batch
from centrex_tlf.lindblad.parameters import Time, square_wave
from centrex_tlf.lindblad.plan_static import prepare_lindblad_problem
from centrex_tlf.lindblad.reference_dense import reference_rhs
from centrex_tlf.lindblad.utils_setup import OBESystem

rust=pytest.importorskip("centrex_tlf.centrex_tlf_rust")
MODE="structured_sparse_dense_diag_soa_single_unpack"
KERNELS=["auto","original","avx2","avx2_fma","avx512","avx512_fma"]


def _prepared(n=9,kernel="original",representation="decomposed",dynamic=True):
    t=sp.Symbol("t",real=True)
    a=sp.Symbol("a",real=True)
    h=sp.diag(*[sp.Float(0.13*i-0.4) for i in range(n)])
    phase=sp.exp(0.7j*t) if dynamic else 1
    for i in range(n-1):
        value=a*phase*(0.2+0.03j*(i+1))
        h[i,i+1]=value
        h[i+1,i]=sp.conjugate(value)
    c=np.zeros((1 if n==1 else 3,n,n),complex)
    c[0,0,0]=np.sqrt(0.13)*1j  # population-preserving dephasing
    if n>1:
        c[1,0,n-1]=np.sqrt(0.3)
        c[2,1,n-1]=np.sqrt(0.17)*1j
    system=OBESystem([],[],[],np.zeros((n,n),complex),np.eye(n),[],h,c,None,[a],[])
    return prepare_lindblad_problem(system,{"a":0.8},cpu_kernel=kernel,
                                    hamiltonian_representation=representation)


def _density(n):
    rng=np.random.default_rng(101+n)
    a=rng.normal(size=(n,n))+1j*rng.normal(size=(n,n))
    a=a@a.conj().T
    return a/np.trace(a)


def _native(payload,kernel):
    data=copy.deepcopy(payload)
    data["cpu_kernel"]=kernel
    try:
        return rust.prepare_lindblad_problem_py(data)
    except ValueError as error:
        if "not supported by this CPU/OS" in str(error):
            return None
        raise


@pytest.mark.parametrize("n",[1,2,4,5,8,9,17])
@pytest.mark.parametrize("representation",["decomposed","entrywise"])
def test_full_complex_rhs_all_supported_kernels_and_vector_tails(n,representation):
    prepared=_prepared(n,representation=representation)
    rho=_density(n)
    y=prepared.layout.pack(rho)
    for kernel in KERNELS:
        plan=_native(prepared.to_payload(),kernel)
        if plan is None:
            continue
        assert plan.cpu_kernel==kernel
        if kernel=="auto":
            assert plan.resolved_cpu_kernel in {"original","avx2","avx2_fma"}
        else:
            assert plan.resolved_cpu_kernel==kernel
        evaluator=rust.create_lindblad_rhs_evaluator_py(plan,MODE)
        for t in [0.0,0.13,0.71,0.13]:
            expected=reference_rhs(prepared,y,t)
            actual=evaluator.rhs_packed_py(y,t)
            np.testing.assert_allclose(actual,expected,atol=3e-13,rtol=3e-13)
            # Matrix and packed calls share cache buffers; alternating APIs must
            # neither reuse stale scratch nor change the original matrix path.
            matrix=evaluator.rhs_matrix_py(rho.ravel(),t).reshape(n,n)
            np.testing.assert_allclose(prepared.layout.pack(matrix),expected,atol=3e-13,rtol=3e-13)
            np.testing.assert_allclose(evaluator.rhs_packed_py(y,t),expected,atol=3e-13,rtol=3e-13)


@pytest.mark.parametrize("dynamic",[False,True])
def test_same_time_overrides_invalidate_hamiltonian_and_sparse_cache(dynamic):
    prepared=_prepared(dynamic=dynamic)
    y=prepared.layout.pack(_density(9))
    original=rust.create_lindblad_rhs_evaluator_py(prepared.rust_plan,MODE)
    slot=prepared.parameter_graph["slot_names"].index("a")
    for kernel in KERNELS:
        plan=_native(prepared.to_payload(),kernel)
        if plan is None:
            continue
        evaluator=rust.create_lindblad_rhs_evaluator_py(plan,MODE)
        for value in [0.0,1.7,-0.3,0.8]:
            for target in [evaluator,original]:
                target.set_scalar_parameter_overrides_py([slot],np.array([value],complex))
            np.testing.assert_allclose(evaluator.rhs_packed_py(y,0.2),original.rhs_packed_py(y,0.2),atol=5e-13,rtol=5e-13)
        evaluator.set_scalar_parameter_overrides_py([],np.array([],complex))
        np.testing.assert_allclose(evaluator.rhs_packed_py(y,0.2),reference_rhs(prepared,y,0.2),atol=5e-13)


@pytest.mark.parametrize("kernel",KERNELS)
@pytest.mark.parametrize("output",["full","populations","selected","photon_rate","photon_integral","full_and_integral"])
def test_single_parallel_batch_and_grid_outputs(kernel,output):
    reference=_prepared()
    if _native(reference.to_payload(),kernel) is None:
        pytest.skip("ISA unavailable")
    candidate=prepare_lindblad_problem(reference.obe_system,{"a":0.8},cpu_kernel=kernel)
    assert candidate.cpu_kernel==kernel
    assert candidate.resolved_cpu_kernel==candidate.rust_plan.resolved_cpu_kernel
    assert candidate.to_payload()["cpu_kernel"]==kernel
    rho=_density(9)
    common=dict(execution_mode=MODE,solver="tsit5",dt=1e-3,abstol=1e-11,reltol=1e-9,
                output=output,output_when="saveat",saveat=[0.0,0.1,0.2])
    if output=="selected":
        common["output_indices"]=[(0,0),(0,3),(3,0)]
    if output in {"photon_rate","photon_integral","full_and_integral"}:
        common["integral_weights"]=[(8,0.47)]
    def values(result):
        return result.packed_y if hasattr(result,"packed_y") else result.values
    before=solve_lindblad(reference,rho,(0.0,0.2),**common)
    after=solve_lindblad(candidate,rho,(0.0,0.2),**common)
    np.testing.assert_allclose(values(after),values(before),atol=2e-11,rtol=2e-9)
    overrides=dict(parameter_slots=["a"],parameter_batch=np.array([[0.3],[1.2]],complex))
    batch=np.stack([rho,rho.conj()])
    before=solve_lindblad_batch(reference,batch,(0.0,0.2),parallel=False,**overrides,**common)
    after=solve_lindblad_batch(candidate,batch,(0.0,0.2),parallel=True,threads=2,**overrides,**common)
    np.testing.assert_allclose(after.values,before.values,atol=2e-11,rtol=2e-9)
    if output not in {"full","full_and_integral"}:
        scan={"a":np.array([0.3,1.2])}
        before=grid_scan(reference,rho,(0.0,0.2),scan=scan,parallel=False,**common)
        after=grid_scan(candidate,rho,(0.0,0.2),scan=scan,parallel=True,threads=2,**common)
        np.testing.assert_allclose(after.values,before.values,atol=2e-11,rtol=2e-9)


def test_auto_payload_default_profile_and_bad_input():
    prepared=_prepared(kernel="auto")
    assert prepared.cpu_kernel=="auto"
    assert prepared.resolved_cpu_kernel in {"original","avx2","avx2_fma"}
    payload=prepared.to_payload()
    del payload["cpu_kernel"]
    plan=rust.prepare_lindblad_problem_py(payload)
    assert plan.cpu_kernel=="auto"
    assert plan.resolved_cpu_kernel==prepared.resolved_cpu_kernel
    evaluator=rust.create_lindblad_rhs_evaluator_py(plan,MODE)
    evaluator.enable_profile_py()
    y=prepared.layout.pack(_density(9))
    evaluator.rhs_packed_py(y,0.13)
    assert evaluator.profile_summary_py()["calls"]==1
    with pytest.raises(ValueError):
        evaluator.rhs_packed_py(y[:-1],0.13)
    with pytest.raises(TypeError,match="cpu_kernel must be a string"):
        prepare_lindblad_problem(object(),{},cpu_kernel=True)
    with pytest.raises(ValueError,match="unsupported cpu_kernel"):
        prepare_lindblad_problem(object(),{},cpu_kernel="fastest")
    payload["cpu_kernel"]="fastest"
    with pytest.raises(ValueError,match="unsupported cpu_kernel"):
        rust.prepare_lindblad_problem_py(payload)


def test_python_preparation_does_not_claim_native_resolution():
    reference=_prepared()
    prepared=prepare_lindblad_problem(reference.obe_system,{"a":0.8},backend="python",cpu_kernel="auto")
    assert prepared.resolved_cpu_kernel is None


def test_switched_drive_and_terminal_event_match_original():
    reference=_prepared()
    params={"a":0.8*square_wave(Time(),2*np.pi*20.0,np.pi/3)}
    original=prepare_lindblad_problem(reference.obe_system,params,cpu_kernel="original")
    automatic=prepare_lindblad_problem(reference.obe_system,params,cpu_kernel="auto")
    rho=_density(9)
    options=dict(solver="tsit5",execution_mode=MODE,dt=1e-4,abstol=1e-11,reltol=1e-9,
                 saveat=np.linspace(0,0.2,21),stop_event=Time()-0.173,collect_stats=True)
    before=solve_lindblad(original,rho,(0.0,0.2),**options)
    after=solve_lindblad(automatic,rho,(0.0,0.2),**options)
    np.testing.assert_allclose(after.t,before.t,atol=1e-10,rtol=0)
    np.testing.assert_allclose(after.packed_y,before.packed_y,atol=3e-9,rtol=1e-7)
    assert after.solver_stats["event_triggered"]
    assert after.t[-1]==pytest.approx(0.173,abs=1e-10)

"""Direct blockwise/SIMD D+S+U trials; isolated snapshots, no library edits."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import pickle
import subprocess
import sys
from pathlib import Path

import bench_rc_propagation as driver
import bench_rust_specialized_rhs as base
import check_density_reduction as reduction
import check_shared_mixtures_sectors as sectors
import confirm_prepared_hamiltonian_integration as integrated
import numpy as np
from check_current_cpu_solvers import base_prefix
from threadpoolctl import threadpool_limits

from centrex_tlf.lindblad.state_layout import PackedHermitianLayout

HERE = Path(__file__).resolve().parent
CACHE = base.REPO / '.cache/benchmarks/block_simd_dsu_v1'
RESULTS = HERE / 'rust_specialized_rhs_results/block_simd_dsu'
ROUNDS = 8
MODES = {'production': 0, 'hooked_control': 0, 'blocks': 1, 'simd': 2, 'both': 3}


def patch_rhs(source):
    source = source.replace('pub struct RhsWorkspace {', '''pub struct RhsWorkspace {
    bench_flags: usize,
    bench_runs: Vec<Vec<(usize,usize)>>,
    bench_support: Vec<bool>,''', 1)
    source = source.replace('upper_layout: upper_layout.clone(),',
                            'bench_flags:0, bench_runs:Vec::new(), bench_support:Vec::new(),\n            upper_layout: upper_layout.clone(),', 1)
    marker = 'if mode.uses_soa_scratch() {\n                add_commutator_sparse_dense_upper_soa('
    assert source.count(marker) == 1
    source = source.replace(marker, '''if workspace.bench_flags != 0 {
                bench_simd_dsu(&plan.hamiltonian_sparse_pattern,
                    &workspace.h_sparse_values, &workspace.hamiltonian_upper,
                    &workspace.rho, &mut workspace.scratch_re, &mut workspace.scratch_im,
                    &workspace.upper_layout, &mut workspace.drho_upper,
                    mode.uses_diag_split(), workspace.bench_flags,
                    &workspace.bench_runs, &workspace.bench_support);
            } else ''' + marker)
    return source + '\n' + (HERE/'block_simd_dsu.rs').read_text()


def build(hooked, module_prefix='_block_simd_dsu_', extra_rust='', exported_functions=()):
    name = module_prefix + ('hooks' if hooked else 'control')
    folder = CACHE/name
    folder.mkdir(parents=True, exist_ok=True)
    rhs_path = base.REPO/'rust/src/lindblad/rhs.rs'
    rhs = rhs_path.read_text()
    rhs = patch_rhs(rhs) if hooked else rhs + '''
pub fn benchmark_configure(_w:&mut RhsWorkspace,_flags:usize,_support:&[usize]) {}
'''
    snapshot = folder/'rhs.rs'
    snapshot.write_text(rhs)
    helper = (HERE/'specialized_rhs_native.rs').read_text()
    imports = helper[:helper.index('struct BenchRhs')]
    imports += 'use lindblad::rhs::benchmark_configure;\n'
    tail = helper[helper.index('#[pyfunction]'):]
    tail = tail.replace('N*N', 'rhs.dim()').replace('GATE_SLOTS', '&rhs.gates')
    tail = tail.replace('=&rhs.gates.iter()', '=rhs.gates.iter()')
    init = 'let mut rhs=BenchRhs::new(plan,mode);'
    assert tail.count(init) == 3
    tail = tail.replace(init, init + '''
    let support:Vec<usize>=payload.get_item("benchmark_support")?.extract()?;
    benchmark_configure(&mut rhs.workspace,mode,&support);
''')
    source = base_prefix().replace(json.dumps(str(rhs_path)), json.dumps(str(snapshot)))
    exports=''.join(f'm.add_function(wrap_pyfunction!({function},m)?)?;\n' for function in exported_functions)
    source += imports + integrated.WRAPPER + tail + extra_rust + f'''
#[pymodule]
fn {name}(m:&Bound<'_,PyModule>)->PyResult<()> {{
    m.add_function(wrap_pyfunction!(rhs_check,m)?)?;
    m.add_function(wrap_pyfunction!(trajectory,m)?)?;{exports}Ok(())
}}
'''
    (folder/'lib.rs').write_text(source)
    (folder/'Cargo.toml').write_text(f'''[package]
name="{name}"
version="0.1.0"
edition="2021"
[lib]
name="{name}"
path="lib.rs"
crate-type=["cdylib"]
[dependencies]
pyo3={{version="0.27",features=["extension-module","num-complex"]}}
numpy="0.27"
num-complex="0.4"
[profile.release]
opt-level=3
lto="fat"
codegen-units=1
''')
    env = dict(os.environ, PYO3_PYTHON=sys.executable, CARGO_TARGET_DIR=str(CACHE/'target'))
    with (folder/'build.log').open('w') as log:
        done = subprocess.run(['cargo','build','--release','--offline','--manifest-path',str(folder/'Cargo.toml')],env=env,stdout=log,stderr=log)
    if done.returncode:
        raise RuntimeError((folder/'build.log').read_text()[-8000:])
    target = folder/(name+'.pyd')
    target.write_bytes((CACHE/'target/release'/(name+'.dll')).read_bytes())
    spec = importlib.util.spec_from_file_location(name,target)
    native = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(native)
    return native, {'wrapper':hashlib.sha256(source.encode()).hexdigest(), 'rhs':hashlib.sha256(rhs.encode()).hexdigest()}


def fixture(name):
    path = CACHE/(name+'_fixture.pkl')
    # Invalidate cached field-dressed matrices when model construction or any
    # package Python source changes, including molecular constants.
    inputs = sorted((base.REPO/'centrex_tlf').rglob('*.py'))
    inputs += [HERE/'bench_rc_propagation.py', HERE/'bench_rust_specialized_rhs.py',
               HERE/'check_shared_mixtures_sectors.py', HERE/'check_density_reduction.py',
               Path(__file__)]
    fingerprint = hashlib.sha256(b''.join(p.read_bytes() for p in inputs)).hexdigest()
    if path.exists():
        cached = pickle.loads(path.read_bytes())
        if isinstance(cached, dict) and cached.get('fingerprint') == fingerprint:
            return cached['data']
    if name == 'rc':
        case = driver.build_case(driver.CONFIGS['nominal'])
        prepared, _ = sectors._prepare_rc_static_gate_plan(case)
        rho0 = sectors._initial_j_mixtures(case)[0][1]
    else:
        prepared,rho0,_ = base.model(name)
    seeds = np.flatnonzero(np.diag(rho0).real>0).tolist()
    support = reduction.coordinate_closure(prepared,seeds)
    data = (prepared.to_payload(),rho0,support)
    path.write_bytes(pickle.dumps({'fingerprint':fingerprint,'data':data}))
    return data


def run():
    CACHE.mkdir(parents=True,exist_ok=True)
    RESULTS.mkdir(parents=True,exist_ok=True)
    print('Building generic current-source controls and kernels',flush=True)
    control,control_hashes = build(False)
    hooks,hook_hashes = build(True)
    if '--build-only' in sys.argv:
        return
    report = {'rounds':ROUNDS,'duration_s':10e-6,'sources':{'control':control_hashes,'hooks':hook_hashes},
              'source_files':{str(p.relative_to(base.REPO)):hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in [Path(__file__), HERE/'block_simd_dsu.rs',
                                        base.REPO/'rust/src/lindblad/eval.rs',
                                        base.REPO/'rust/src/lindblad/plan.rs',
                                        base.REPO/'rust/src/ode/tsit5.rs']},'systems':{}}
    def save():
        (RESULTS/'results.json').write_text(json.dumps(report,indent=2))
    for name in ['rc','r2']:
        print(name+': preparing fixture',flush=True)
        payload,rho0,support = fixture(name)
        n = len(rho0)
        layout = PackedHermitianLayout(n)
        payload['benchmark_support'] = sorted(i*n+j for i,j in support)
        y0 = layout.pack(rho0)
        rng = np.random.default_rng(238+n)
        max_rhs = 0.0
        # Arbitrary compatible coherences as well as evolving physical states.
        for _ in range(3):
            rho = reduction.random_positive_density(n,support,rng)
            y = layout.pack(rho)
            for t in np.array([125,375,625,875])*1e-9:
                expected = np.asarray(control.rhs_check(payload,y,float(t),0))
                for mode in [0,1,2,3]:
                    actual = np.asarray(hooks.rhs_check(payload,y,float(t),mode))
                    error = float(np.max(abs(actual-expected))/max(1,float(np.max(abs(expected)))))
                    max_rhs = max(max_rhs,error)
                    assert error < 1e-10,(name,mode,t,error)
        reference = layout.unpack(np.asarray(control.trajectory(payload,y0,0,10e-6,1e-13,1e-11)[0]))
        row = {'n':n,'support_coordinates':len(support),'max_normalized_rhs_error':max_rhs,'samples':{key:[] for key in MODES},'orders':[]}
        report['systems'][name] = row
        def solve(key, payload=payload, y0=y0, layout=layout, reference=reference, name=name):
            native = control if key=='production' else hooks
            result = native.trajectory(payload,y0,MODES[key],10e-6,1e-11,1e-9)
            rho = layout.unpack(np.asarray(result[0]))
            accuracy = base.accuracy(rho,reference)
            trace = float(abs(np.trace(rho)-1))
            assert accuracy['passed'] and trace<1e-9,(name,key,accuracy,trace)
            return {'seconds':result[1],'rhs_calls':result[2],'accepted':result[3],'rejected':result[4],'accuracy':accuracy,'trace_error':trace}
        for key in MODES:
            solve(key)
        for repeat in range(ROUNDS):
            order = list(MODES)
            rng.shuffle(order)
            row['orders'].append(order)
            print(f'{name}: round {repeat+1}/{ROUNDS}',flush=True)
            for key in order:
                row['samples'][key].append(solve(key))
            save()
    save()
    print('Complete',flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=1):
        run()

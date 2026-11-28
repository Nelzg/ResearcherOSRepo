"""Independent dense numerical check of the formulas; does not execute MATLAB."""
import json
from pathlib import Path
import numpy as np

rng = np.random.default_rng(20261128)
T, r, N, M = 7, 3, 8, 4

def complex_random(shape):
    return rng.normal(size=shape) + 1j * rng.normal(size=shape)

Y = complex_random((T, M))
model = [.3 * complex_random((T, r)), .3 * complex_random((r, r)), .3 * complex_random((r, N))]
mask = rng.random((N, M)) > .25
mask[:, -1] = False
weights = .1 + rng.random((N, M))
penalties = [.1 + rng.random(x.shape) for x in model]

def evaluate(factors):
    R, S, C = factors
    Q = R @ S @ C
    E = np.exp(2*np.pi*(Q.imag-Q.imag.max(axis=0))) * np.exp(-2j*np.pi*Q.real)
    F = E / np.linalg.norm(E, axis=0)
    Z = np.zeros((N, M), complex)
    for k in range(M):
        a = mask[:, k]
        A = F[:, a]
        Z[a, k] = np.linalg.solve(A.conj().T @ A + np.diag(weights[a, k]), A.conj().T @ Y[:, k])
    residual = Y-F@Z
    f = -np.vdot(Y, F@Z).real + sum(np.sum(p*np.abs(x)**2) for p,x in zip(penalties,factors))
    GF = -2*residual@Z.conj().T
    GQ = 2j*np.pi*(GF*F.conj()-np.abs(F)**2*np.sum((GF.conj()*F).real,axis=0))
    gradients = [GQ@(S@C).conj().T, R.conj().T@GQ@C.conj().T, (R@S).conj().T@GQ]
    gradients = [g+2*p*x for g,p,x in zip(gradients,penalties,factors)]
    return f, gradients, np.linalg.norm(residual)/np.linalg.norm(Y), Z

_, gradients, nme, z = evaluate(model)
errors=[]
for k in range(3):
    for real_direction in (False,True):
        D=complex_random(model[k].shape)
        if real_direction: D=D.real
        D=D/np.linalg.norm(D)
        plus=[x.copy() for x in model]; minus=[x.copy() for x in model]
        h=1e-6; plus[k]+=h*D; minus[k]-=h*D
        observed=(evaluate(plus)[0]-evaluate(minus)[0])/(2*h)
        predicted=np.vdot(gradients[k],D).real
        errors.append(float(abs(observed-predicted)/max(1,abs(observed),abs(predicted))))
assert max(errors)<1e-7
assert np.all(z[~mask]==0)
assert nme<=1
evidence={"scope":"Independent NumPy dense-system formula check, not MATLAB execution", "numpy":np.__version__,
          "gradient_relative_errors":errors,"maximum_gradient_error":max(errors),"NME":float(nme),"passed":True}
Path(__file__).with_name('gradient_reference_evidence.json').write_text(json.dumps(evidence,indent=2)+'\n',encoding='utf-8')
print(json.dumps(evidence,indent=2))

function x = nlcr_constraints(action,model,o,x)
%NLCR_CONSTRAINTS Tangent projection, retraction, and feasibility checks.
if strcmp(action,'project')
    names={'R','S','C'};
    for k=1:3
        name=names{k};
        if o.real(k) || (o.block && k~=2), x.(name)=real(x.(name)); end
        if ~o.train(k), x.(name)=zeros(size(x.(name)),'like',x.(name)); end
    end
    x.S=structure(x.S,model,o,false);
    if o.orthogonalR
        x.R=x.R-model.R*hermitian(model.R'*x.R);
    elseif o.unitR
        x.R=x.R-model.R.*sum(real(conj(model.R).*x.R),1);
    end
    if o.orthogonalC
        x.C=x.C-hermitian(x.C*model.C')*model.C;
    elseif o.unitC
        x.C=x.C-model.C.*sum(real(conj(model.C).*x.C),2);
    end
elseif strcmp(action,'retract')
    x=model;
    names={'R','S','C'};
    for k=1:3
        name=names{k};
        if o.real(k) || (o.block && k~=2), x.(name)=real(x.(name)); end
    end
    x.S=structure(x.S,x,o,true);
    if o.orthogonalR, x.R=polar(x.R);
    elseif o.unitR, x.R=normalizeColumns(x.R); end
    if o.orthogonalC, x.C=polar(x.C')';
    elseif o.unitC, x.C=normalizeColumns(x.C')'; end
elseif strcmp(action,'check')
    x=true; tol=o.constraintTolerance;
    assert(size(model.R,2)==size(model.S,1) && size(model.S,2)==size(model.C,1), ...
        'nlcr:dimensions','Factor dimensions do not agree.');
    names={'R','S','C'};
    for k=1:3
        A=model.(names{k}); assert(all(isfinite(A(:))),'nlcr:nonfinite','Nonfinite factor.');
        if o.real(k) || (o.block && k~=2)
            assert(norm(imag(A),'fro')<=tol,'nlcr:constraints','A real factor is complex.');
        end
    end
    assert(norm(model.S-structure(model.S,model,o,true),'fro')<=tol,'nlcr:constraints','S structure violated.');
    if o.orthogonalR
        assert(norm(model.R'*model.R-eye(size(model.R,2)),'fro')<=tol,'nlcr:constraints','R is not orthonormal.');
    elseif o.unitR
        assert(all(abs(sum(abs(model.R).^2,1)-1)<=tol),'nlcr:constraints','R column norms violated.');
    end
    if o.orthogonalC
        assert(norm(model.C*model.C'-eye(size(model.C,1)),'fro')<=tol,'nlcr:constraints','C is not orthonormal.');
    elseif o.unitC
        assert(all(abs(sum(abs(model.C).^2,2)-1)<=tol),'nlcr:constraints','C row norms violated.');
    end
else
    error('nlcr:action','Unknown constraint action.');
end
end

function S=structure(S,m,o,isPoint)
if o.block
    phase=(m.rowBlock(:)==1) & (m.colBlock(:).'==1);
    amplitude=(m.rowBlock(:)==2) & (m.colBlock(:).'==2);
    S=real(S).*phase+1i*imag(S).*amplitude;
    if o.real(2), S=real(S); end
end
if o.diagonalS, S=S.*(m.rowIds(:)==m.colIds(:).'); end
if o.fixedIdentityS
    assert(size(S,1)==size(S,2),'nlcr:constraints','Identity S must be square.');
    S=zeros(size(S),'like',S);
    if isPoint, S=eye(size(S)); end
end
end

function A=polar(A)
assert(size(A,1)>=size(A,2),'nlcr:constraints','Too few coordinates for an orthonormal basis.');
if isempty(A), return; end
[U,~,V]=svd(A,'econ'); A=U*V';
end

function A=normalizeColumns(A)
n=sqrt(sum(abs(A).^2,1));
assert(all(n>0),'nlcr:constraints','Cannot normalize a zero vector. Supply another initialization or pruning policy.');
A=A./n;
end

function A=hermitian(A)
A=(A+A')/2;
end

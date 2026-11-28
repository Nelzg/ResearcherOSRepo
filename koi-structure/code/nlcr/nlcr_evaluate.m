function [value,Z,g,info,F] = nlcr_evaluate(Y,model,weights,mask,o,needGradient)
%NLCR_EVALUATE Objective, masked PCG and total derivatives at frozen weights.
if nargin<6, needGradient=true; end
F=nlcr_dictionary(model);
assert(all(isfinite(weights.Z(:))) && all(weights.Z(:)>0), ...
    'nlcr:weights','Coefficient weights must remain finite and positive.');
Z=zeros(size(mask),'like',Y+1i);
info=struct('solves',0,'cgIterations',0,'maxResidual',0,'energy',0);
for m=1:size(Y,2)
    active=find(mask(:,m));
    if isempty(active), continue; end
    A=F(:,active); w=weights.Z(active,m); b=A'*Y(:,m);
    op=@(v) A'*(A*v)+w.*v;
    pre=@(v) v./(sum(abs(A).^2,1).'+w);
    if norm(b)==0
        u=zeros(size(b),'like',b); flag=0; iter=0;
    else
        [u,flag,~,iter]=pcg(op,b,o.cgTolerance,o.cgMaxIterations,pre,[],zeros(size(b),'like',b));
    end
    residual=norm(op(u)-b)/max(norm(b),realmin);
    assert(flag==0 && isfinite(residual) && residual<=max(10*o.cgTolerance,1e-12), ...
        'nlcr:cg','PCG did not converge (flag %d, residual %.3g).',flag,residual);
    Z(active,m)=u;
    info.solves=info.solves+1; info.cgIterations=info.cgIterations+iter;
    info.maxResidual=max(info.maxResidual,residual);
end
res=Y-F*Z;
info.energy=norm(res,'fro')^2+sum(weights.Z(:).*abs(Z(:)).^2);
% Numerical certificate used for final NME and all accepted evaluations.
assert(info.energy<=norm(Y,'fro')^2*(1+max(100*o.cgTolerance,1e-10)), ...
    'nlcr:cg','PCG solution failed the regularized-energy certificate.');
value=-real(sum(conj(Y(:)).*reshape(F*Z,[],1)));
names={'R','S','C'};
for k=1:3
    X=model.(names{k}); W=weights.(names{k});
    value=value+sum(W(:).*abs(X(:)).^2);
end
assert(isfinite(value),'nlcr:nonfinite','Nonfinite objective.');
g=[];
if ~needGradient, return; end
% Includes dZ/dF: f = min_Z (||Y-FZ||^2 + Z'*W*Z) - ||Y||^2.
GF=-2*res*Z';
columnCorrection=sum(real(conj(GF).*F),1);
GQ=2*pi*1i*(GF.*conj(F)-abs(F).^2.*columnCorrection);
g.R=GQ*(model.S*model.C)';
g.S=model.R'*GQ*model.C';
g.C=(model.R*model.S)'*GQ;
for k=1:3
    name=names{k}; g.(name)=g.(name)+2*weights.(name).*model.(name);
end
end

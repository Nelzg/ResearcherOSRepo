function [model,Z,note] = nlcr_initialize(T,M,o)
%NLCR_INITIALIZE Local RNG; QR -> SVD preserves Q when constraints permit.
stream=RandStream('mt19937ar','Seed',o.seed);
d=o.r*(1+o.block);
model=struct('R',[],'S',[],'C',[],'rowIds',1:d,'colIds',1:d, ...
    'rowBlock',ones(1,d),'colBlock',ones(1,d),'atomIds',1:o.N);
if o.block
    model.rowBlock(o.r+1:end)=2; model.colBlock(o.r+1:end)=2;
end
note='Requested initialization; no replacement.';
if ~isempty(o.initialModel)
    supplied=o.initialModel;
    for name={'R','S','C'}
        assert(isfield(supplied,name{1}),'nlcr:initialization','initialModel needs R,S,C.');
        model.(name{1})=supplied.(name{1});
    end
    assert(isequal(size(model.R),[T d]) && isequal(size(model.S),[d d]) ...
        && isequal(size(model.C),[d o.N]),'nlcr:initialization','Wrong initial model dimensions.');
    nlcr_constraints('check',model,o);
    note='User-supplied feasible initial model; preserved exactly.';
else
    fourier=ismember(o.initialization,{'DFT','SemiDFT'});
    if fourier
        if strcmp(o.initialization,'DFT')
            assert(o.r==1,'nlcr:initialization','DFT requires r=1.');
            axesT=T; axesN=o.N;
        else
            axesT=o.axesT; axesN=o.axesN;
            validateAxes(axesT,T,o.r); validateAxes(axesN,o.N,o.r);
        end
        model.R=coordinates(axesT); model.S=eye(o.r);
        model.C=(coordinates(axesN)./reshape(axesN,1,[])).';
        if o.block
            model.R=[model.R randn(stream,T,o.r)];
            model.C=[model.C;randn(stream,o.r,o.N)];
            model.S=blkdiag(model.S,1i*zeros(o.r));
        end
    else
        model.R=sample(stream,T,d,o.initialization);
        model.S=sample(stream,d,d,o.initialization);
        model.C=sample(stream,d,o.N,o.initialization);
        if o.block
            model.R=real(model.R); model.C=real(model.C);
            model.S=blkdiag(real(model.S(1:o.r,1:o.r)),1i*real(model.S(o.r+1:end,o.r+1:end)));
        end
    end
    original=model;
    try
        if o.block || o.fixedIdentityS
            % Absorption would violate block/identity constraints.
            model=nlcr_constraints('retract',model,o);
        else
            if o.orthogonalR || o.unitR
                if o.orthogonalR
                    assert(T>=d,'nlcr:constraints','r exceeds T.');
                    [U,B]=qr(model.R,0); model.R=U; model.S=B*model.S;
                else
                    norms=sqrt(sum(abs(model.R).^2,1));
                    assert(all(norms>0),'nlcr:constraints','Zero column.');
                    model.R=model.R./norms; model.S=diag(norms)*model.S;
                end
            end
            if o.orthogonalC || o.unitC
                if o.orthogonalC
                    assert(o.N>=d,'nlcr:constraints','r exceeds N.');
                    [U,B]=qr(model.C.',0); model.C=U.'; model.S=model.S*B.';
                else
                    norms=sqrt(sum(abs(model.C).^2,2));
                    assert(all(norms>0),'nlcr:constraints','Zero row.');
                    model.C=model.C./norms; model.S=model.S*diag(norms);
                end
            end
            if o.diagonalS
                [U,D,V]=svd(model.S); model.R=model.R*U; model.S=D; model.C=V'*model.C;
            end
            nlcr_constraints('check',model,o);
        end
    catch err
        if ~startsWith(err.identifier,'nlcr:'), rethrow(err); end
        % Mixed real constraints may not survive complex QR/SVD absorption.
        model=nlcr_constraints('retract',original,o);
        note=['Replaced by direct feasible projection: ' err.message];
    end
    nlcr_constraints('check',model,o);
    q0=original.R*original.S*original.C;
    q1=model.R*model.S*model.C;
    if norm(q1-q0,'fro')>1e-10*max(1,norm(q0,'fro'))
        note=[note ' Constraints changed initial Q; exact Fourier preservation is not claimed.'];
    end
end
if isempty(o.initialZ)
    Z=complex(zeros(o.N,M));
else
    Z=o.initialZ;
    assert(isequal(size(Z),[o.N M]) && all(isfinite(Z(:))),'nlcr:initialization','Invalid initialZ.');
end
end

function A=sample(stream,a,b,name)
switch name
    case 'Uniform', A=rand(stream,a,b);
    case 'Randn', A=randn(stream,a,b);
    case 'ComplexUniform', A=complex(rand(stream,a,b),rand(stream,a,b));
    case 'ComplexRandn', A=complex(randn(stream,a,b),randn(stream,a,b));
end
end

function validateAxes(a,total,r)
assert(isvector(a) && numel(a)==r && isreal(a) && all(isfinite(a)) ...
    && all(a>=1 & a==fix(a)) && prod(a)==total,'nlcr:axes', ...
    'Axis sizes must be r positive integers with the requested product.');
end

function R=coordinates(a)
R=zeros(prod(a),numel(a)); index=(0:prod(a)-1).'; stride=1;
for k=1:numel(a)
    R(:,k)=mod(floor(index/stride),a(k)); stride=stride*a(k);
end
end

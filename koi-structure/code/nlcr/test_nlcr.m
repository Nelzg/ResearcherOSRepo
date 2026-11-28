function evidence = test_nlcr()
%TEST_NLCR Small deterministic numerical checks; not the channel experiment.
rng(20261128); T=7; N=8; M=3; r=3;
Y=complex(randn(T,M),randn(T,M));
base=struct('N',N,'r',r,'maxIterations',3,'cgTolerance',1e-12, ...
    'lambda',[.03 .02 .03 .04],'reweighted',[1 1 1 1], ...
    'quadratic',[0 1 1 1],'tau',[.01 0 0 0]);
o=nlcr_options(base); [m,z]=nlcr_initialize(T,M,o);
% Moderate scale avoids a nearly saturated amplitude map in derivative tests.
m.R=.2*m.R; m.S=.2*m.S; m.C=.2*m.C;
w=nlcr_weights(m,z,o); mask=true(N,M); mask(1:2,2)=false; mask(:,3)=false;
[~,z,g,info,F]=nlcr_evaluate(Y,m,w,mask,o);
assert(all(z(~mask)==0)); assert(max(abs(sum(abs(F).^2,1)-1))<1e-13);
gradientErrors=zeros(1,3); names={'R','S','C'};
for k=1:3
    name=names{k}; D=complex(randn(size(m.(name))),randn(size(m.(name)))); D=D/norm(D,'fro');
    h=1e-6; plus=m; minus=m; plus.(name)=m.(name)+h*D; minus.(name)=m.(name)-h*D;
    fp=nlcr_evaluate(Y,plus,w,mask,o,false); fm=nlcr_evaluate(Y,minus,w,mask,o,false);
    G=g.(name); predicted=real(sum(conj(G(:)).*D(:))); observed=(fp-fm)/(2*h);
    gradientErrors(k)=abs(predicted-observed)/max([1 abs(predicted) abs(observed)]);
end
assert(max(gradientErrors)<1e-6,'Total complex gradients disagree with finite differences.');
% Compare masked PCG with an independently assembled dense system.
directError=0;
for col=1:M
    a=find(mask(:,col)); if isempty(a), continue; end
    A=F(:,a); direct=(A'*A+diag(w.Z(a,col)))\(A'*Y(:,col));
    directError=max(directError,norm(direct-z(a,col))/max(1,norm(direct)));
end
assert(directError<1e-10);
% Large imaginary offset must not overflow, and is a column gauge symmetry.
gauge=struct('R',ones(T,1),'S',1,'C',1i*[1e5 -1e5]);
Fg=nlcr_dictionary(gauge); assert(all(isfinite(Fg(:))));
assert(norm(Fg-ones(T,2)/sqrt(T),'fro')<1e-12);
% QR -> SVD preserves complex Q and both orthogonality constraints.
qbase=base; qbase.orthogonalR=true; qbase.orthogonalC=true;
qbase.diagonalS=true; qbase.prunePolicy='paired';
qopt=nlcr_options(qbase); [qm,~,qnote]=nlcr_initialize(T,M,qopt);
[original,~]=nlcr_initialize(T,M,o);
preservation=norm(original.R*original.S*original.C-qm.R*qm.S*qm.C,'fro')/norm(original.R*original.S*original.C,'fro');
assert(preservation<1e-12); nlcr_constraints('check',qm,qopt);
% SemiDFT ordering: first axis changes fastest.
fo=nlcr_options(struct('N',12,'r',2,'initialization','SemiDFT','axesT',[2 3],'axesN',[3 4]));
[fm,~]=nlcr_initialize(6,M,fo); FF=nlcr_dictionary(fm);
F1=exp(-2*pi*1i*(0:1)'*(0:2)/3)/sqrt(2);
F2=exp(-2*pi*1i*(0:2)'*(0:3)/4)/sqrt(3);
fourierError=norm(FF-kron(F2,F1),'fro'); assert(fourierError<1e-12);
% Both algorithms and all optimizer variants, with line-search monotonicity.
variants={'None','AdamW','Muon'}; tested=0;
for algorithm=1:2
    for k=1:numel(variants)
        cfg=base; cfg.algorithm=algorithm; cfg.optimizer=variants{k};
        cfg.innerMaxIterations=3; cfg.initialModel=m;
        out=nlcr_fit(Y,cfg); assert(out.success,out.failureReason);
        assert(out.iterations==3 && out.NME<=1+1e-10 && all(out.Z(~out.mask)==0));
        assert(all([out.history.objectiveAfterStep]<=[out.history.objectiveBeforeStep]));
        tested=tested+1;
    end
end
% The final weights must be updated from the state AFTER the last pruning.
cfg=struct('N',N,'r',1,'initialization','DFT','train',[false false false], ...
    'maxIterations',1,'cgTolerance',1e-12);
oo=nlcr_options(cfg); [mm,z0]=nlcr_initialize(T,M,oo); mk=true(N,M);
ww=nlcr_weights(mm,z0,oo); [~,z1]=nlcr_evaluate(Y,mm,ww,mk,oo,false);
[mm,z1,mk]=nlcr_prune(mm,z1,mk,oo);
ww=nlcr_weights(mm,z1,oo); [~,expected]=nlcr_evaluate(Y,mm,ww,mk,oo,false);
out=nlcr_fit(Y,cfg); assert(out.success && norm(out.Z-expected,'fro')<1e-12);
assert(nnz(out.Z)>0,'Initial zero Z was incorrectly locked.');
% Pruned coefficient is never resurrected; exact threshold equality survives.
zz=zeros(N,M); zz(1,1)=1; zz(2,1)=.01; zz(3,1)=.009;
[mm,zz,mk]=nlcr_prune(original,zz,true(N,M),o);
assert(size(zz,1)==2 && mk(2,1) && ~mk(2,2));
[~,zz]=nlcr_evaluate(Y,mm,nlcr_weights(mm,zz,o),mk,o,false);
assert(all(zz(~mk)==0));
% Entirely zero representation, zero maximum, and strict final failure threshold.
cfg.tau=[2 0 0 0]; out=nlcr_fit(Y,cfg);
assert(out.success && out.activeRows==0 && out.NME==1 && isnan(out.correlation));
cfg.tau=[0 0 0 0]; cfg.maxComponents=0; out=nlcr_fit(Y,cfg);
assert(~out.success && out.activeRows>0 && isfinite(out.NME));
% Constraints and structural pruning under explicitly selected policies.
cases={struct('real',[true true true],'unitR',true,'unitC',true), ...
    struct('orthogonalR',true,'orthogonalC',true), ...
    struct('diagonalS',true,'prunePolicy','paired'), ...
    struct('fixedIdentityS',true,'prunePolicy','paired'), ...
    struct('block',true,'r',2,'prunePolicy','blockwise'), ...
    struct('block',true,'r',2,'prunePolicy','blockwise','real',[1 1 1]), ...
    struct('block',true,'r',2,'prunePolicy','blockwise','orthogonalR',true,'orthogonalC',true)};
for j=1:numel(cases)
    cfg=base; fields=fieldnames(cases{j});
    for k=1:numel(fields), cfg.(fields{k})=cases{j}.(fields{k}); end
    out=nlcr_fit(Y,cfg); assert(out.success,out.failureReason);
    nlcr_constraints('check',out.model,out.options); tested=tested+1;
end
% Rectangular core and block labels survive independent row/column deletion.
mm=original; mm.S(1,:)=0; mm.S(:,2)=0; zz=ones(N,M);
[mm,~,~]=nlcr_prune(mm,zz,true(N,M),o);
assert(isequal(size(mm.S),[r-1 r-1]) && isequal(mm.rowIds,[2 3]) && isequal(mm.colIds,[1 3]));
mm=original; mm.S(1,:)=0;
[mm,~,~]=nlcr_prune(mm,zz,true(N,M),o); assert(isequal(size(mm.S),[r-1 r]));
% Invalid split is rejected rather than inferred.
caught=false;
try
    bad=fo; bad.axesT=[2 2]; nlcr_initialize(6,M,bad);
catch err
    caught=strcmp(err.identifier,'nlcr:axes');
end
assert(caught);
% Aggregation excludes failures; zero reconstruction only invalidates correlation.
c1=struct('N',4,'r',1,'initialization','DFT','train',[0 0 0],'maxIterations',1,'tau',[2 0 0 0]);
c2=c1; c2.tau=[0 0 0 0]; c2.maxComponents=0;
summary=nlcr_series(@(seed) Y+seed*1e-3,{c1,c2},[1 2]);
assert(summary.meanNME(1)==1 && isnan(summary.meanNME(2)));
assert(isequal(summary.successfulCount,[2;0]) && all(summary.correlationCount==0));
evidence=struct('gradientRelativeErrors',gradientErrors,'denseSolveRelativeError',directError, ...
    'QR_SVD_relativeError',preservation,'SemiDFT_absoluteError',fourierError, ...
    'maxPCGResidual',info.maxResidual,'fitVariantsChecked',tested,'initializationNote',qnote, ...
    'matlabVersion',version,'passed',true);
disp(evidence);
end

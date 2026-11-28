function o = nlcr_options(user)
%NLCR_OPTIONS Numerical defaults, not an approved experiment configuration.
o = struct('algorithm',1,'maxIterations',64,'N',[],'r',[], ...
    'initialization','ComplexRandn','seed',1,'axesT',[],'axesN',[], ...
    'lambda',[.01 0 0 0],'epsilon',[.01 .01 .01 .01], ...
    'reweighted',[true false false false],'quadratic',[false false false false], ...
    'tau',[.01 0 0 0],'initialZ',[],'initialModel',[], ...
    'real',[false false false],'unitR',false,'unitC',false, ...
    'orthogonalR',false,'orthogonalC',false,'diagonalS',false, ...
    'fixedIdentityS',false,'block',false,'train',[true true true], ...
    'optimizer','None','learningRate',1,'beta1',.9,'beta2',.999, ...
    'optimizerEpsilon',1e-8,'weightDecay',0,'cgTolerance',1e-10, ...
    'cgMaxIterations',1000,'maxBacktracks',30,'backtrackFactor',.5, ...
    'innerMaxIterations',20,'innerTolerance',1e-6,'maxComponents',100, ...
    'maxNME',2,'constraintTolerance',1e-9,'prunePolicy','ordinary');
if nargin == 0, return; end
assert(isstruct(user) && isscalar(user),'nlcr:options','Options must be a scalar struct.');
names = fieldnames(user);
for k=1:numel(names)
    assert(isfield(o,names{k}),'nlcr:options','Unknown option: %s',names{k});
    o.(names{k}) = user.(names{k});
end
validateattributes(o.N,{'numeric'},{'scalar','integer','positive','finite'});
validateattributes(o.r,{'numeric'},{'scalar','integer','positive','finite'});
for name = {'maxIterations','cgMaxIterations','maxBacktracks','innerMaxIterations'}
    validateattributes(o.(name{1}),{'numeric'},{'scalar','integer','positive','finite'});
end
assert(ismember(o.algorithm,[1 2]),'nlcr:options','algorithm must be 1 or 2.');
for name = {'lambda','epsilon','tau'}
    validateattributes(o.(name{1}),{'numeric'},{'real','finite','nonnegative','vector','numel',4});
    o.(name{1}) = reshape(o.(name{1}),1,4);
end
assert(all(o.epsilon>0) && o.lambda(1)>0,'nlcr:options','Positive epsilon and lambda(1) are required.');
for name = {'reweighted','quadratic'}
    x=o.(name{1}); assert(numel(x)==4 && all(ismember(x(:),[0 1])),'nlcr:options','Expected four switches.');
    o.(name{1})=logical(reshape(x,1,4));
end
assert(o.reweighted(1) && ~o.quadratic(1),'nlcr:options','Z uses only its specified reweighted system.');
for name = {'real','train'}
    x=o.(name{1}); assert(numel(x)==3 && all(ismember(x(:),[0 1])),'nlcr:options','Expected three switches.');
    o.(name{1})=logical(reshape(x,1,3));
end
for name = {'unitR','unitC','orthogonalR','orthogonalC','diagonalS','fixedIdentityS','block'}
    x=o.(name{1}); assert(isscalar(x) && ismember(x,[0 1]),'nlcr:options','Expected scalar switch.');
end
for name = {'learningRate','optimizerEpsilon','cgTolerance','innerTolerance','constraintTolerance'}
    validateattributes(o.(name{1}),{'numeric'},{'real','finite','positive','scalar'});
end
for name = {'beta1','beta2','backtrackFactor'}
    x=o.(name{1}); assert(isscalar(x) && isreal(x) && isfinite(x) && x>0 && x<1,'nlcr:options','Expected value in (0,1).');
end
validateattributes(o.weightDecay,{'numeric'},{'real','finite','nonnegative','scalar'});
validateattributes(o.maxComponents,{'numeric'},{'integer','nonnegative','scalar','finite'});
validateattributes(o.maxNME,{'numeric'},{'real','nonnegative','scalar','finite'});
validateattributes(o.seed,{'numeric'},{'integer','nonnegative','scalar','finite'});
o.optimizer=validatestring(o.optimizer,{'None','AdamW','Muon'});
o.initialization=validatestring(o.initialization,{'Uniform','Randn','ComplexUniform','ComplexRandn','DFT','SemiDFT'});
o.prunePolicy=validatestring(o.prunePolicy,{'ordinary','paired','blockwise'});
assert(~o.block || ~o.fixedIdentityS,'nlcr:constraints','Block S cannot equal the identity.');
if o.fixedIdentityS, o.train(2)=false; end
if o.block
    assert(strcmp(o.prunePolicy,'blockwise'),'nlcr:policy','Explicitly select prunePolicy=blockwise for block models.');
elseif o.diagonalS || o.fixedIdentityS
    assert(strcmp(o.prunePolicy,'paired'),'nlcr:policy','Explicitly select prunePolicy=paired for diagonal/identity S.');
else
    assert(strcmp(o.prunePolicy,'ordinary'),'nlcr:policy','Use ordinary pruning for an unstructured model.');
end
end

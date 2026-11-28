function result = nlcr_fit(Y,user)
%NLCR_FIT Two reweighted nonlinear channel representation algorithms.
% result=nlcr_fit(Y,options). See README.md for explicitly chosen policies.
o=nlcr_options(user);
validateattributes(Y,{'double'},{'2d','nonempty','finite'});
assert(norm(Y,'fro')>0,'nlcr:data','A zero channel has undefined NME.');
[model,Z,note]=nlcr_initialize(size(Y,1),size(Y,2),o);
mask=true(size(Z)); state=[]; total=emptyCost(); history=struct([]);
reason=''; completed=0; started=tic;
try
    for q=1:o.maxIterations
        weights=nlcr_weights(model,Z,o);
        [value,Z,g,cost]=nlcr_evaluate(Y,model,weights,mask,o);
        total=addCost(total,cost); innerCount=0;
        if o.algorithm==2 && o.train(2)
            [model,value,Z,g,cost,innerCount]=coreSolve(Y,model,weights,mask,o);
            total=addCost(total,cost);
        end
        stepOptions=o;
        if o.algorithm==2, stepOptions.train(2)=false; end
        tangent=nlcr_constraints('project',model,stepOptions,g);
        [direction,pending]=optimizerDirection(model,tangent,state,stepOptions);
        [candidate,newValue,newZ,accepted,alpha,cost,ninner]=searchStep( ...
            Y,model,value,Z,direction,weights,mask,o,stepOptions);
        total=addCost(total,cost); innerCount=innerCount+ninner;
        if accepted, model=candidate; Z=newZ; state=pending; end
        beforePruneValue=newValue;
        [model,Z,mask,event]=nlcr_prune(model,Z,mask,o);
        if event.resized, state=[]; end % No stale moments after dimension changes.
        completed=q;
        history(q)=struct('iteration',q,'objectiveBeforeStep',value, ...
            'objectiveAfterStep',beforePruneValue,'accepted',accepted,'step',alpha, ...
            'activeRowsAfterPrune',nnz(any(Z~=0,2)),'atoms',size(Z,1), ...
            'latentRows',size(model.S,1),'latentColumns',size(model.S,2), ...
            'innerIterations',innerCount,'pruning',event);
    end
    % One extra weight update and masked solve; no further pruning.
    weights=nlcr_weights(model,Z,o);
    [~,Z,~,cost,F]=nlcr_evaluate(Y,model,weights,mask,o,false);
    total=addCost(total,cost);
catch err
    if ~startsWith(err.identifier,'nlcr:'), rethrow(err); end
    reason=[err.identifier ': ' err.message]; F=[];
end
if isempty(reason)
    estimate=F*Z; nme=norm(Y-estimate,'fro')/norm(Y,'fro');
    K=nnz(any(Z~=0,2)); correlation=NaN;
    if norm(estimate,'fro')>0
        correlation=min(1,abs(sum(conj(Y(:)).*estimate(:)))/(norm(Y,'fro')*norm(estimate,'fro')));
    end
    if K>o.maxComponents, reason='Too many active rows after the final solve.'; end
    if nme>o.maxNME
        reason=strtrim([reason ' Final NME exceeds the configured threshold.']);
    end
else
    % Failed numerical runs never expose stale state as a final reconstruction.
    estimate=[]; nme=NaN; correlation=NaN; K=NaN;
end
result=struct('model',model,'Z',Z,'mask',mask,'F',F,'estimate',estimate, ...
    'NME',nme,'correlation',correlation,'activeRows',K,'success',isempty(reason), ...
    'failureReason',reason,'iterations',completed,'history',history, ...
    'cost',total,'seconds',toc(started),'initializationNote',note,'options',o);
end

function [m,value,Z,g,total,count]=coreSolve(Y,m,w,mask,o)
% Algorithm 2: warm-start from the same current S for every outer candidate.
inner=o; inner.train=[false o.train(2) false]; inner.optimizer='None';
[value,Z,g,c]=nlcr_evaluate(Y,m,w,mask,o); total=addCost(emptyCost(),c); count=0;
for k=1:o.innerMaxIterations
    tg=nlcr_constraints('project',m,inner,g);
    if norm(tg.S,'fro')<=o.innerTolerance*max(1,norm(m.S,'fro')), break; end
    d=tg; d.R=-d.R; d.S=-d.S; d.C=-d.C;
    plain=o; plain.algorithm=1;
    [candidate,newValue,~,accepted,~,c]=searchStep(Y,m,value,Z,d,w,mask,plain,inner);
    total=addCost(total,c);
    if ~accepted, break; end
    improvement=value-newValue; m=candidate; count=count+1;
    [value,Z,g,c]=nlcr_evaluate(Y,m,w,mask,o); total=addCost(total,c);
    if improvement<=o.innerTolerance*max(1,abs(value)), break; end
end
end

function [best,bestValue,bestZ,accepted,alpha,total,ninner]=searchStep(Y,m,value,Z,d,w,mask,o,stepOptions)
best=m; bestValue=value; bestZ=Z; accepted=false; alpha=0; total=emptyCost(); ninner=0;
if norm(d.R,'fro')+norm(d.S,'fro')+norm(d.C,'fro')==0, return; end
for attempt=1:o.maxBacktracks
    a=o.learningRate*o.backtrackFactor^(attempt-1);
    candidate=m; candidate.R=m.R+a*d.R; candidate.S=m.S+a*d.S; candidate.C=m.C+a*d.C;
    try
        candidate=nlcr_constraints('retract',candidate,stepOptions);
        nlcr_constraints('check',candidate,stepOptions);
        if o.algorithm==2 && o.train(2)
            [candidate,v,z,~,c,count]=coreSolve(Y,candidate,w,mask,o);
            ninner=ninner+count;
        else
            [v,z,~,c]=nlcr_evaluate(Y,candidate,w,mask,o,false);
        end
        total=addCost(total,c);
    catch err
        if ~startsWith(err.identifier,'nlcr:'), rethrow(err); end
        total.rejectedNumericalCandidates=total.rejectedNumericalCandidates+1;
        continue;
    end
    if v<=value
        best=candidate; bestValue=v; bestZ=z; accepted=true; alpha=a; return;
    end
end
end

function [d,next]=optimizerDirection(m,g,state,o)
names={'R','S','C'};
if isempty(state)
    state.t=0;
    for k=1:3
        name=names{k}; state.m.(name)=zeros(size(g.(name)),'like',g.(name));
        state.v.(name)=zeros(size(g.(name)));
    end
end
next=state; next.t=state.t+1; d=g;
for k=1:3
    name=names{k}; G=g.(name);
    switch o.optimizer
        case 'None'
            D=-G;
        case 'AdamW'
            next.m.(name)=o.beta1*state.m.(name)+(1-o.beta1)*G;
            next.v.(name)=o.beta2*state.v.(name)+(1-o.beta2)*abs(G).^2;
            mh=next.m.(name)/(1-o.beta1^next.t); vh=next.v.(name)/(1-o.beta2^next.t);
            D=-mh./(sqrt(vh)+o.optimizerEpsilon)-o.weightDecay*m.(name);
        case 'Muon'
            next.m.(name)=o.beta1*state.m.(name)+(1-o.beta1)*G;
            A=next.m.(name);
            if isempty(A) || norm(A,'fro')==0
                D=zeros(size(A),'like',A);
            else
                % Complex polar momentum, exact SVD, zero singular directions omitted.
                [U,S,V]=svd(A,'econ'); s=diag(S); keep=s>max(size(A))*eps(max(s));
                D=-U(:,keep)*V(:,keep)'-o.weightDecay*m.(name);
            end
    end
    d.(name)=D;
end
d=nlcr_constraints('project',m,o,d);
slope=0;
for k=1:3
    name=names{k}; G=g.(name); D=d.(name);
    slope=slope+real(sum(conj(G(:)).*D(:)));
end
% Adaptive moments can point uphill for the actual frozen objective.
if slope>=0
    for k=1:3, name=names{k}; d.(name)=-g.(name); end
end
end

function a=emptyCost()
a=struct('evaluations',0,'solves',0,'cgIterations',0,'maxResidual',0,'rejectedNumericalCandidates',0);
end

function a=addCost(a,b)
if isfield(b,'evaluations'), a.evaluations=a.evaluations+b.evaluations;
else, a.evaluations=a.evaluations+1; end
a.solves=a.solves+b.solves; a.cgIterations=a.cgIterations+b.cgIterations;
a.maxResidual=max(a.maxResidual,b.maxResidual);
if isfield(b,'rejectedNumericalCandidates')
    a.rejectedNumericalCandidates=a.rejectedNumericalCandidates+b.rejectedNumericalCandidates;
end
end

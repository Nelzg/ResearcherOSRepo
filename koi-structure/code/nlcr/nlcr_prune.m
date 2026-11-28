function [model,Z,mask,event] = nlcr_prune(model,Z,mask,o)
%NLCR_PRUNE Permanent Z mask and dimension-consistent structural removal.
before=model; oldMask=mask;
if o.reweighted(1)
    Z=threshold(Z,o.tau(1));
    mask=mask & (Z~=0); % Initial zeros are free; only this operation locks them.
end
names={'R','S','C'};
for k=1:3
    % Fixed factors are never modified by elementwise thresholding.
    if o.reweighted(k+1) && o.train(k)
        model.(names{k})=threshold(model.(names{k}),o.tau(k+1));
    end
end
keepAtoms=any(Z~=0,2);
Z=Z(keepAtoms,:); mask=mask(keepAtoms,:);
model.C=model.C(:,keepAtoms); model.atomIds=model.atomIds(keepAtoms);
keepRows=any(model.S~=0,2); keepCols=any(model.S~=0,1);
if o.diagonalS || o.fixedIdentityS
    % Preserve matching latent identifiers (and square shape for paired policy).
    live=intersect(model.rowIds(keepRows),model.colIds(keepCols));
    keepRows=ismember(model.rowIds,live); keepCols=ismember(model.colIds,live);
end
model.R=model.R(:,keepRows); model.S=model.S(keepRows,keepCols);
model.C=model.C(keepCols,:);
model.rowIds=model.rowIds(keepRows); model.colIds=model.colIds(keepCols);
model.rowBlock=model.rowBlock(keepRows); model.colBlock=model.colBlock(keepCols);
if isempty(model.atomIds)
    % Empty representation: remove unused latent dimensions as well.
    model.R=zeros(size(model.R,1),0); model.S=zeros(0,0); model.C=zeros(0,0);
    model.rowIds=[]; model.colIds=[]; model.rowBlock=[]; model.colBlock=[];
else
    % Explicit policy: restore constraints, even if this changes Q after pruning.
    % Factor zeros are not permanent; only the coefficient mask is permanent.
    model=nlcr_constraints('retract',model,o);
end
nlcr_constraints('check',model,o);
event=struct('newForbidden',nnz(oldMask)-nnz(mask), ...
    'removedAtoms',numel(before.atomIds)-numel(model.atomIds), ...
    'removedRows',numel(before.rowIds)-numel(model.rowIds), ...
    'removedColumns',numel(before.colIds)-numel(model.colIds), ...
    'resized',~isequal(size(before.S),size(model.S)) || ~isequal(size(before.C),size(model.C)));
end

function X=threshold(X,tau)
if isempty(X), return; end
scale=max(abs(X(:)));
if scale>0, X(abs(X)/scale<tau)=0; end
% A zero maximum leaves X identically zero; never divide 0 by 0.
end

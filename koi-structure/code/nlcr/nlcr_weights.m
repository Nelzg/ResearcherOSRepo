function w = nlcr_weights(model,Z,o)
%NLCR_WEIGHTS Freeze these arrays for an entire outer iteration.
w.Z=o.lambda(1)./(abs(Z).^2+o.epsilon(1));
names={'R','S','C'};
for k=1:3
    X=model.(names{k}); j=k+1;
    w.(names{k})=o.reweighted(j)*o.lambda(j)./(abs(X).^2+o.epsilon(j)) ...
        +o.quadratic(j)*o.lambda(j)*ones(size(X));
end
end

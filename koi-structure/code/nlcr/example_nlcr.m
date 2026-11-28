function result = example_nlcr(Y,algorithm)
%EXAMPLE_NLCR Fit supplied data. Does not run GenRealisticChannel automatically.
if nargin<2, algorithm=1; end
options=struct('N',256,'r',2,'algorithm',algorithm, ...
    'initialization','Randn','real',[true true true], ...
    'maxIterations',64,'lambda',[.01 0 0 0]);
result=nlcr_fit(Y,options);
end

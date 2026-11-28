function summary = nlcr_series(generator,configurations,seeds,outputFile)
%NLCR_SERIES Explicit invocation only. Never silently replaces a generator.
% configurations: cell array of option structs; seeds defaults to 1:50.
% Successful NME means exclude failed runs. Correlation has its own denominator.
if nargin<3, seeds=1:50; end
if nargin<4, outputFile=''; end
assert(isa(generator,'function_handle'),'nlcr:series','Pass a channel generator function handle.');
assert(iscell(configurations) && ~isempty(configurations),'nlcr:series','Expected nonempty cell configurations.');
validateattributes(seeds,{'numeric'},{'vector','integer','finite','nonnegative','nonempty'});
assert(numel(unique(seeds))==numel(seeds),'nlcr:series','Channel seeds must be unique.');
A=numel(configurations); B=numel(seeds);
for a=1:A, nlcr_options(configurations{a}); end
records=cell(A,B); nme=nan(A,B); corr=nan(A,B); successful=false(A,B);
for s=1:B
    % Generate once so all configurations see identical Y, including stochastic generators.
    try
        Y=generator(seeds(s)); generationError='';
    catch err
        generationError=[err.identifier ': ' err.message];
    end
    for a=1:A
        if isempty(generationError)
            try
                out=nlcr_fit(Y,configurations{a});
            catch err
                out=struct('success',false,'failureReason',[err.identifier ': ' err.message], ...
                    'NME',NaN,'correlation',NaN);
            end
        else
            out=struct('success',false,'failureReason',['Generator: ' generationError], ...
                'NME',NaN,'correlation',NaN);
        end
        out.channelSeed=seeds(s); out.configurationIndex=a;
        records{a,s}=out;
        successful(a,s)=out.success && isfinite(out.NME);
        if successful(a,s), nme(a,s)=out.NME; corr(a,s)=out.correlation; end
    end
    if ~isempty(outputFile)
        save(outputFile,'records','configurations','seeds','successful','nme','corr','-v7.3');
    end
end
means=nan(A,1); corrMeans=nan(A,1); corrCount=sum(isfinite(corr),2);
for a=1:A
    if any(successful(a,:)), means(a)=mean(nme(a,successful(a,:))); end
    valid=isfinite(corr(a,:)); if any(valid), corrMeans(a)=mean(corr(a,valid)); end
end
paired=nan(A,A); pairCounts=zeros(A,A);
for a=1:A
    for b=1:A
        common=successful(a,:) & successful(b,:); pairCounts(a,b)=nnz(common);
        if any(common), paired(a,b)=mean(nme(a,common)-nme(b,common)); end
    end
end
summary=struct('records',{records},'seeds',seeds,'meanNME',means, ...
    'successfulCount',sum(successful,2),'failureCount',B-sum(successful,2), ...
    'meanCorrelation',corrMeans,'correlationCount',corrCount, ...
    'pairedNMEDifference',paired,'pairCounts',pairCounts);
if ~isempty(outputFile), save(outputFile,'summary','-append'); end
end

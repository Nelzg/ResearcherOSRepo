function [w, U] = cpd_core_l2(T, U0, alpha)
%CPD_CORE_L2 Fit a real dense tensor with L2-regularized CP weights.
%   [w,U] = cpd_core_l2(T,cpd(T,R),alpha) returns a row of component
%   weights and unit-norm columns in each factor matrix. alpha > 0.
%   Minimizes (||T - Tfit||_F^2 + alpha * ||w||_2^2)/2 locally by ALS.

validateattributes(alpha, {'numeric'}, {'scalar','real','finite','positive'});
U = U0;
R = size(U{1}, 2);
order = numel(U);
grams = cell(size(U));
for n = 1:order
    norms = vecnorm(U{n}, 2, 1);
    U{n}(1, norms == 0) = 1;
    norms(norms == 0) = 1;
    U{n} = U{n} ./ norms;
    grams{n} = U{n}' * U{n};
end

lastObjective = Inf;
dataNormSq = norm(T(:))^2;
for sweep = 1:500
    for n = 1:order
        % Absorb w into mode n. Its squared Frobenius norm is sum(w.^2).
        G = ones(R, R);
        for k = [1:n-1, n+1:order]
            G = G .* grams{k};
        end
        M = mtkrprod(T, U, n);
        B = M / (G + alpha * eye(R));
        w = vecnorm(B, 2, 1);
        active = w > 0;
        U{n}(:, active) = B(:, active) ./ w(active);
        grams{n} = U{n}' * U{n};
    end

    % At the block solution, the penalized objective simplifies to this.
    objective = 0.5 * max(0, dataNormSq - sum(B(:) .* M(:)));
    if abs(lastObjective - objective) <= 1e-10 * max(objective, eps(dataNormSq))
        return
    end
    lastObjective = objective;
end
warning('cpd_core_l2:MaxSweeps', 'Returned the fit after 500 ALS sweeps.');
end

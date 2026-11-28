function F = nlcr_dictionary(model)
%NLCR_DICTIONARY Stable column-normalized exp(-2*pi*1i*R*S*C).
Q=model.R*model.S*model.C;
assert(all(isfinite(Q(:))),'nlcr:nonfinite','Nonfinite R*S*C.');
% Subtract before multiplying by 2*pi: no overflow of positive amplitudes.
shifted=imag(Q)-max(imag(Q),[],1);
E=exp(2*pi*shifted).*exp(-2*pi*1i*mod(real(Q),1));
F=E./sqrt(sum(abs(E).^2,1));
assert(all(isfinite(F(:))),'nlcr:nonfinite','Nonfinite normalized dictionary.');
end

function[bet] = basis_function(S,c,Z)
 bet = exp(-((S-c)'*inv(Z)*(S-c))/2);
end
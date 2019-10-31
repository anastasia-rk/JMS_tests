function[volume] = ellipsoid_volume(k, alph, C)
chisquare = chi2inv(1-alph,k);
eigenvals = eig(C);
term1 = 2*pi^(k/2)/(k*gamma(k/2));
term2 = chisquare^(k/2);
term3 = geomean(eigenvals)^(k/2);
volume = term1*term2*term3;
end
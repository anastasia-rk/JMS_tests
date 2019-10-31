function[X_el,Y_el,Z_el] = build_ellipsoid_from_data(triple_theta)
s = 4.28; % s = 1.96 for 72% confidence. for 95% use s=4.28, for 98% s = 7.815
alpha = [0:pi/10:2*pi];
beta = [0:pi/10:2*pi];

[COEFF, SCORE, LATENT, TSQUARED, EXPLAINED, MU] = pca(triple_theta);
 db = sqrt(s*LATENT);

for i=1:length(alpha)
    for j=1:length(beta)
        p(1,1) = db(1)*cos(alpha(i))*cos(beta(j));
        p(2,1) = db(2)*sin(alpha(i))*cos(beta(j));
        p(3,1) = db(3)*sin(beta(j));
        % rotation
        r = COEFF*p;
        X_el(i,j) = r(1) + MU(1);
        Y_el(i,j) = r(2) + MU(2);
        Z_el(i,j) = r(3) + MU(3);
    end
end

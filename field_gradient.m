function[grad] = field_gradient(S,Z,knots,basis)
L = size(knots,2);
% Compute gradient of chemotactic field for current cell position
switch basis
    case 'gaussian'
        for i=1:L
           % gradient for gaussian functions is calculated analitically
           c = knots(:,i);
           BB = basis_function(S,c,Z);
           Z_inv = inv(Z);
           grad(1,i) = -(basis_function(S,c,Z)*(2*Z_inv(1,1)*(S(1,1) - c(1,1)) + (Z_inv(1,2) + Z_inv(2,1)*(S(2,1) - c(2,1))) )/2);
           grad(2,i) = -(basis_function(S,c,Z)*(2*Z_inv(2,2)*(S(2,1) - c(2,1)) + (Z_inv(1,2) + Z_inv(2,1)*(S(1,1) - c(1,1))) )/2);
        end
    case 'bspline'
        % gradient for spline function is calculated numerically
        delta = 0.0000001;
        Sd = S + delta;
        index = 1;
        for i=1:2:L-1
           support_border_x = knots(1,i:i+1);
           support_border_y = knots(2,i:i+1);
           coef_x = (support_border_x(2)-support_border_x(1))/4;
           coef_y = (support_border_y(2)-support_border_y(1))/4;
           grad(1,index) = (biorthogonal_spline(Sd(1)/coef_x,S(2)/coef_y,support_border_x/coef_x,support_border_y/coef_y) - biorthogonal_spline(S(1)/coef_x,S(2)/coef_y,support_border_x/coef_x,support_border_y/coef_y))/delta;
           grad(2,index) = (biorthogonal_spline(S(1)/coef_x,Sd(2)/coef_y,support_border_x/coef_x,support_border_y/coef_y) - biorthogonal_spline(S(1)/coef_x,S(2)/coef_y,support_border_x/coef_x,support_border_y/coef_y))/delta;
           index = index + 1;
        end
        nn = 1;
end

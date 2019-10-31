function[out] = dynfun(x,A,B,theta,Z,knots,basis_type)
beta = field_gradient(x(1:2),Z,knots,basis_type);
out = A*x + B*beta*theta;
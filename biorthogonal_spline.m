function[basis_function] = biorthogonal_spline(x,y,support_border_x,support_border_y)

N3x = cubic_spline(x,support_border_x);
N3y = cubic_spline(y,support_border_y);

basis_function = N3x*N3y;
end
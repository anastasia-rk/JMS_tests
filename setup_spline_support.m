function[knots] = setup_spline_support(grid_limits,nx,ny)
% NOT FINISHED
% for radial basis functions knots are centers of bf
knot_nx = nx + 3;
knot_ny = ny + 3;
dx = (grid_limits(3)- grid_limits(1))/(knot_nx);
dy = (grid_limits(4)- grid_limits(2))/(knot_ny);

for i=1:knot_nx+1
    x_knot(i) = grid_limits(1) + dx*(i-1);
end
for i=1:knot_ny+1
    y_knot(i) = grid_limits(2) + dy*(i-1);
end
k = 1;
for i=1:knot_nx-3
    for j=1:knot_ny-3
        knots(1,k) = x_knot(i);
        knots(1,k+1) = x_knot(i+4);
        knots(2,k) = y_knot(j);
        knots(2,k+1) = y_knot(j+4);
        k = k + 2;
    end
end

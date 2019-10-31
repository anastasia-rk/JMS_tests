function[plotted,ZZ] = plot_gradient(Theta,Z,knots,grid_limits,basis_type)
dx = (grid_limits(3) - grid_limits(1))/100;
dy = (grid_limits(4) - grid_limits(2))/100;
coordinate_x = [grid_limits(1):dx:grid_limits(3)];
coordinate_y = [grid_limits(2):dy:grid_limits(4)];
N = length(coordinate_x);
M = length(coordinate_y);
ll = size(knots,2);

[X_grid, Y_grid] = meshgrid(coordinate_x,coordinate_y);

[X_grid, Y_grid] = meshgrid(coordinate_x,coordinate_y);

switch basis_type
    case 'gaussian'
        ll = size(knots,2);
        for i = 1:N
            for j = 1:M
                for index1 = 1:ll
                   S = [coordinate_x(i); coordinate_y(j)];
                   c = knots(:,index1);
                   pow = (S - c)'*inv(Z)*((S - c));
                   z(j,i,index1) = Theta(index1)*exp(-pow/2);   
                end
                ZZ(j,i) = sum(z(j,i,:));
            end
        end
    case 'bspline'
         ll = size(knots,2) - 1;
         for i = 1:N
            for j = 1:M
                index1 = 1;
                for index = 1:2:ll 
                   support_x = knots(1,index:index+1);
                   support_y = knots(2,index:index+1);
                   coef_x = (support_x(2)-support_x(1))/4;
                   coef_y = (support_y(2)-support_y(1))/4;
                   bf = biorthogonal_spline(coordinate_x(i)/coef_x,coordinate_y(j)/coef_y,support_x/coef_x,support_y/coef_y);
                   z(j,i,index1) = Theta(index1)*bf;   
                   index1 = index1 + 1;
                end
                ZZ(j,i) = sum(z(j,i,:));
            end
        end
end
[dx, dy] = gradient(ZZ, 10, 10);
Z_min = min(ZZ);
Z_min_min = min(Z_min);
% if Z_min_min < 0 
%     ZZ = ZZ + abs(Z_min_min)*ones(M,N);
% end
contour(X_grid, Y_grid, ZZ), hold on
quiver(X_grid, Y_grid, dx, dy, 1, 'k');


plotted = true;
end
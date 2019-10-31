function[Theta] = initiate_field(body,wound,limits,knots)
dx = (limits(3) - limits(1))/10;
dy = (limits(4) - limits(2))/8;
coordinate_x = [limits(1):dx:limits(3)];
coordinate_y = [limits(2):dy:limits(4)];
N = length(coordinate_x);
M = length(coordinate_y);

% z = ax + by + c;
c = (body*limits(3) - wound*limits(1))/(limits(1)+limits(3));
a = (wound - c)/limits(3);
b = 0;

% z = A*theta

% generate vector z
for i=1:N
    for j=1:M
        Z((i-1)*M + j,1) = a*coordinate_x(i) + b*coordinate_y(j) + c;
    end
end

% generate matrix A
ll = size(knots,2) - 1;
for i=1:N
    for j=1:M
        k = 0;
        for index = 1:2:ll 
            support_x = knots(1,index:index+1);
            support_y = knots(2,index:index+1);
            coef_x = (support_x(2)-support_x(1))/4;
            coef_y = (support_y(2)-support_y(1))/4;
            k = k + 1;
            A((i-1)*M + j,k) = biorthogonal_spline(coordinate_x(i)/coef_x,coordinate_y(j)/coef_y,support_x/coef_x,support_y/coef_y);
        end
    end
end

% Least square estimate
 Theta = inv(A'*A)*A'*Z;
end
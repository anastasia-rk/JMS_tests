function[N3x] = cubic_spline(x,support_border)
% adjust the support
if (support_border(1)>0)
    x = x - support_border(1);
    support_border = support_border - support_border(1);
end
dx = (support_border(2) - support_border(1))/4;
for i=1:5
    support(i) = support_border(1) + dx*(i-1);
end
% compute scaling function
if     (x>=support(1)) && (x<=support(2))
    N3x = x^3;
elseif (x>support(2)) && (x<=support(3))
    N3x = 4 - 12*x + 12*x^2 - 3*x^3;
elseif (x>support(3)) && (x<=support(4))
    N3x = -44 + 60*x - 24*x^2 + 3*x^3;
elseif (x>support(4)) && (x<=support(5))
    N3x = 64 - 48*x + 12*x^2 - x^3;
else
    N3x = 0;  
end

N3x = N3x/4;
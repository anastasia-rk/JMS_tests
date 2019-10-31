function[ellips_points] = build_ellips(mea,co)
s = 5.991; % s = 5.991 for 95% confidence. for 99% use s=9.210, for 90% s = 4.605
alpha = [0:pi/16:2*pi];
[vb,db] = eig(co);
db = sqrt(s*db);
for i = 1:length(alpha)
    x = [cos(alpha(i)); sin(alpha(i))];
    y = vb*db*x;
    ellips_points(i,1) = y(1) + mea(1);
    ellips_points(i,2) = y(2) + mea(2);
end
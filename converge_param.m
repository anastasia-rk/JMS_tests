function[converged,delta_theta] = converge_param(Theta_c,Theta_old,iter)
% checking if LL is converged by with standard criterion
epsilon = 0.001;

if iter == 1
    delta_theta = 1;
else
    delta_theta = ((Theta_c - Theta_old)'*(Theta_c - Theta_old))/(Theta_c'*Theta_c);
end

if delta_theta < epsilon && iter > 2
    converged = true;
else
    converged = false;
end

end
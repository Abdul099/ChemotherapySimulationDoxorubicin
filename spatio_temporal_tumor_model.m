L = 0.004;
x = linspace(0,L,100);
t = linspace(0,36000,100);
m = 1;
sol = pdepe(m,@tumor_pde,@tumor_ic,@tumor_bc,x,t);
surf(x,t,sol)
xlabel('x(m)')
ylabel('t(s)')
zlabel('drug concentration')
function [c,f,s] = tumor_pde(x,t,u,dudx)
c = 1;
f = 5*10.^(-11)*dudx;
s = u*(-0.5*10.^(-4));
end
function u0 = tumor_ic(x)
u0 = 0;
end
function [pl,ql,pr,qr] = tumor_bc(xl,ul,xr,ur,t)
pl = 0;
ql = 0;
pr = ur-200;
qr = 0;
end

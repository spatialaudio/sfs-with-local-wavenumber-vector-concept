%inspired from
%https://github.com/JensAhrens/soundfieldsynthesis/blob/master/Chapter_4/Fig_4_15.m

clear all
clc
tic

wavelength = 1 %# in m
c = 343;  %# speed of sound in m/s

far = 100;

kr0 = far*pi  %# very large for valid far/hf approx
k = 2*pi/wavelength;  %# in rad/m
f =  c/wavelength;  %# frequency in Hz
omega = 2*pi*f;  %# rad/s
r0 = kr0/k  %# radius of spherical/circular array in m
M = 2*ceil(kr0);  %# number of modes
L = M*2  %# number of secondary sources, >=M*2 to avoid spatial aliasing

%%
phi_0 = [0:L-1]*2*pi/L;
x_0 = r0*cos(phi_0);
y_0 = r0*sin(phi_0);

% 2.5D WFS driving function of plane wave with referencing to origin
x_ref = r0/2;
phi_pw = pi/2;
D = - sqrt(8*pi*1i*k*x_ref) .*  cos(phi_0-phi_pw) .*...
    exp(-1i*k*r0 .* cos(phi_0-phi_pw));
% window valid for phi_pw = pi/2!!!
D( find( phi_0 < pi ) ) = 0;                          

% alloc RAM
D_WFS = zeros(1, 2*M+1);
D_HOA = zeros(1, 2*M+1);

% get Fourier series coefficients:
for m = -M:+M
    D_WFS(1, m+M+1) = 1/(2*pi) * sum(D .* exp(-1i*m.*phi_0),2) * (2*pi)/L;
    D_HOA(1, m+M+1) = 2*1i/kr0 * (-1i)^abs(m) ...
        / (sph_bessel_1st(kr0,abs(m)) - 1i*sph_bessel_2nd(kr0,abs(m))) .*...
        exp(-1i*m*(phi_pw));  % 2.5D HOA driving function plane wave
end
% avoid log of 0
D_WFS(D_WFS==0) = 10^(-300/20);
D_HOA(D_HOA==0) = 10^(-300/20);

% normalize
% TBD check how we get this offset between both approaches:
D_HOA = D_HOA / sqrt(2);

%%
m = (-M:+M) / M;  % we plot over normalized m

subplot(221)
plot(m,real(D_WFS)), hold on
plot(m,real(D_HOA)), hold off
xlabel("m / M, disrcete m!")
ylabel("Re(D[m])")
legend('WFS numericFS', 'HOA analyticFS')
grid on

subplot(222)
plot(m, imag(D_WFS)), hold on
plot(m, imag(D_HOA)), hold off
xlabel("m / M, disrcete m!")
ylabel("Im(D[m])")
grid on

subplot(223)
plot(m, db(D_WFS)), hold on
plot(m, db(D_HOA)), hold off
ylim([-100 20])
xlabel("m / M, disrcete m!")
ylabel("|D[m]| in dB")
grid on

subplot(224)
plot(m, abs(D_WFS)), hold on
plot(m, abs(D_HOA)), hold off
xlabel("m / M, disrcete m!")
ylabel("|D[m]|")
title(['WFS: ', num2str(D_WFS(M+1)), ',   HOA: ', num2str(D_HOA(M+1))])
grid on

toc
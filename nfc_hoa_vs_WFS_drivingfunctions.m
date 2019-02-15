% NFC-HOA vs. WFS driving functions equivalence?!
% inspired from
% https://github.com/JensAhrens/soundfieldsynthesis/blob/master/Chapter_4/Fig_4_15.m
% Frank Schultz, github: fs446, 2019-02-14

clear all
clc
tic

% since the simulation works in normalized kr-domain, we only need to play 
% with the far factor
far = 10;

%%
kr0 = far*pi  % very large for valid far/hf approx
M = 5*ceil(kr0);  % number of modes
L = M*4  % number of secondary sources, >=M*2 to avoid spatial aliasing

wavelength = 1;  % in m, value can be set arbitrarily
c = 343;  % speed of sound in m/s, value can be set arbitrarily

k = 2*pi/wavelength;  % in rad/m
f =  c/wavelength;  % frequency in Hz
omega = 2*pi*f;  % rad/s
r0 = kr0/k;  % radius of spherical/circular array in m

%%
% alloc RAM for Fourier series:
D_WFS_FS_numeric = zeros(1, 2*M+1);
D_HOA_FS_analytic = zeros(1, 2*M+1);

D_WFS_FS_J_numeric = zeros(1, 2*M+1);
D_WFS_FS_J_analytic = zeros(1, 2*M+1);

D_WFS_FS_Sinc_numeric = zeros(1, 2*M+1);
D_WFS_FS_Sinc_analytic = zeros(1, 2*M+1);

D_WFS_FS_Conv_numeric = zeros(1, 2*M+1);
D_WFS_FS_Conv_analytic = zeros(1, 2*M+1);

%%
phi_pw = 4*pi/4;  % we might change the prop direction of plane wave
% note: all handling of sec src sel etc. is hopefully correctly performed
% relativel to phi_pw

phi_start = phi_pw + pi/2;
phi_0 = phi_start + [0:L-1]*2*pi/L;
x_0 = r0*cos(phi_0);
y_0 = r0*sin(phi_0);

phi_int = [0:L-1]*2*pi/L;
idxpi = find(phi_int==pi);

% 2.5D WFS driving function of plane wave with referencing to origin
x_ref = r0/2;
D_WFS = - sqrt(8*pi*1i*k*x_ref) .*  cos(phi_0-phi_pw) .*...
    exp(-1i*k*r0 .* cos(phi_0-phi_pw));
D_WFS(L/2+1:end) = 0;  %sec src sel = spatial window

% get first Fourier series coefficients
for m = -M:+M
    D_WFS_FS_numeric(1, m+M+1) = 1/(2*pi) * ...
        sum(D_WFS .* exp(-1i*m.*phi_0),2) * (2*pi)/L; % numeric WFS
    D_HOA_FS_analytic(1, m+M+1) = 2*1i/kr0 * (-1i)^abs(m) ...
        / (sph_bessel_1st(kr0,abs(m)) - 1i*sph_bessel_2nd(kr0,abs(m))) .*...
        exp(-1i*m*(phi_pw));  % analytic 2.5D HOA driving function plane wave
end

% avoid log of 0
D_WFS_FS_numeric(D_WFS_FS_numeric==0) = 10^(-300/20);
D_HOA_FS_analytic(D_HOA_FS_analytic==0) = 10^(-300/20);

% normalize
% TBD check how we get this offset between both approaches:
D_HOA_FS_analytic = D_HOA_FS_analytic / sqrt(2);

%%
%we check if the contribution of this integral is zero for all m
%although not strictly proven we can assume this so far
if 1
    D_sin1 = zeros(1, 2*M+1);
    D_sin2 = zeros(1, 2*M+1);
    for m=-M:+M
        D_sin1(1, m+M+1) = 1i/(2*pi) * ...
            sum(sin(-(m-1)*(phi_0-phi_pw)) .*...
            exp(1i*-kr0*cos(phi_0-phi_pw))) * (2*pi)/L;
        D_sin2(1, m+M+1) = 1i/(2*pi) *...
            sum(sin(-(m+1)*(phi_0-phi_pw)) .*...
            exp(1i*-kr0*cos(phi_0-phi_pw))) * (2*pi)/L;
    end
    max(abs(real(D_sin1))),max(abs(imag(D_sin1))),
    max(abs(real(D_sin2))),max(abs(imag(D_sin2)))
    clear D_sin1
    clear D_sin2
end
%%
D_WFS_partJ = cos(phi_0-phi_pw) .* exp(-1i*k*r0 .* cos(phi_0-phi_pw));
D_WFS_partSinc = phi_0*0+1;
D_WFS_partSinc(L/2+1:end) = 0;  % truncation window

%tmp = D_WFS_FS_Sinc_analytic*0;

for m = -M:+M
    D_WFS_FS_J_numeric(1, m+M+1) = 1/(2*pi) *...
        sum(D_WFS_partJ .* exp(-1i*m.*phi_0),2) * (2*pi)/L;
    
    D_WFS_FS_J_analytic(1, m+M+1) = exp(-1i*m*phi_pw) / (2*1i^(m-1)) * ...
        (besselj(m-1,kr0) - besselj(m+1,kr0));
    
    D_WFS_FS_Sinc_numeric(1, m+M+1) = 1/(2*pi) *...
        sum(D_WFS_partSinc .* exp(-1i*m.*phi_0),2) * (2*pi)/L;
    
    D_WFS_FS_Sinc_analytic(1, m+M+1) = -1i*(exp(-1i*m*(phi_pw+pi/2)) ...
        - exp(-1i*m*(phi_pw+3*pi/2)))  / m /2/pi;
    if m==0
        D_WFS_FS_Sinc_analytic(1, m+M+1) = +pi /2/pi;
    end
    
    % % for phi_pw=0
    %D_WFS_FS_Sinc_analytic(1, m+M+1) = -1/2 * sin(m*pi/2) / (m*pi/2);
    %if m==0
    %    D_WFS_FS_Sinc_analytic(1, m+M+1) = 1/2;
    %end
    
    % % for phi_pw=pi
    %D_WFS_FS_Sinc_analytic(1, m+M+1) = 1/2 * sin(m*pi/2) / (m*pi/2);
    %if m==0
    %    D_WFS_FS_Sinc_analytic(1, m+M+1) = 1/2;
    %end
    
end

% convolution of Fourier Series
ConvFS_Numeric = - sqrt(8*pi*1i*k*x_ref) *...
    conv(D_WFS_FS_J_numeric,D_WFS_FS_Sinc_numeric);

ConvFS_Numeric = - sqrt(8*pi*1i*k*x_ref) *...
    conv(D_WFS_FS_J_analytic,D_WFS_FS_Sinc_analytic);

idxConv = (length(ConvFS_Numeric)-1)/2-M:(length(ConvFS_Numeric)-1)/2+M;
ConvFS_Numeric = ConvFS_Numeric(idxConv);

%%
% get plots HOA analytic vs. WFS numeric
m = (-M:+M) / ceil(kr0);  % we plot over normalized m

subplot(321)
plot(m, real(D_HOA_FS_analytic)), hold on
plot(m, real(D_WFS_FS_numeric))
plot(m, real(ConvFS_Numeric))

hold off
xlim([-3 +3])
xlabel("m / ceil(kr0), discrete m!")
ylabel("Re(D[m])")
legend('HOA analytic', 'WFS numeric', 'Conv numeric')
grid on


subplot(322)
plot(m, imag(D_HOA_FS_analytic)), hold on
plot(m, imag(D_WFS_FS_numeric))
plot(m, imag(ConvFS_Numeric))

hold off
xlim([-3 +3])
xlabel("m / ceil(kr0), discrete m!")
ylabel("Im(D[m])")
grid on


subplot(323)
plot(m, abs(D_HOA_FS_analytic)), hold on
plot(m, abs(D_WFS_FS_numeric))
plot(m, abs(ConvFS_Numeric))

hold off
xlim([-3 +3])
xlabel("m / ceil(kr0), discrete m!")
ylabel("|D[m]|")
title(['m=0 -> WFS: ', num2str(D_WFS_FS_numeric(M+1)), ...
    ',   HOA: ', num2str(D_HOA_FS_analytic(M+1))])
grid on


subplot(324)
plot(m, db(D_HOA_FS_analytic)), hold on
plot(m, db(D_WFS_FS_numeric))
plot(m, db(ConvFS_Numeric))

hold off
xlim([-3 +3])
ylim([-100 20])
xlabel("m / ceil(kr0), discrete m!")
ylabel("|D[m]| in dB")
grid on


subplot(325)
plot(m,+1/2+real(D_WFS_FS_Sinc_analytic),'LineWidth',2), hold on
plot(m,+1/2+real(D_WFS_FS_Sinc_numeric))
plot(m,-1/2+imag(D_WFS_FS_Sinc_analytic),'LineWidth',2)
plot(m,-1/2+imag(D_WFS_FS_Sinc_numeric))

hold off
xlim([-3 +3])
xlabel("m / ceil(kr0), discrete m!")
ylabel(" ")
grid on


subplot(326)
plot(m,+1/2+real(D_WFS_FS_J_analytic),'LineWidth',2), hold on
plot(m,+1/2+real(D_WFS_FS_J_numeric))
plot(m,-1/2+imag(D_WFS_FS_J_analytic),'LineWidth',2)
plot(m,-1/2+imag(D_WFS_FS_J_numeric))

hold off
xlim([-3 +3])
xlabel("m / ceil(kr0), discrete m!")
ylabel(" ")
grid on

%%
if 0
    figure
    plot(m,db(D_WFS_FS_J_numeric)), hold on
    plot(m,db(D_WFS_FS_Sinc_numeric)), hold off
    hold off
    xlim([-3 +3])
    ylim([-80 20])
    xlabel("m / ceil(kr0), discrete m!")
    ylabel(" ")
    grid on
end

toc
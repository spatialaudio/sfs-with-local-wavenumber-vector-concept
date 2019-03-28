# NFC-HOA vs. WFS driving functions equivalence?!
# https://github.com/spatialaudio/sfs-with-local-wavenumber-vector-concept
# /blob/master/nfc_hoa_vs_WFS_drivingfunctions.py
# Frank Schultz, github: fs446
#
# inspired from
# https://github.com/JensAhrens/soundfieldsynthesis/blob/master/Chapter_4
# /Fig_4_15.m


import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from scipy.special import jv, spherical_jn, spherical_yn



# taken from the SFS toolbox
# https://github.com/sfstoolbox/sfs-python/blob/master/sfs/util.py
def spherical_hn2(n, z):
    r"""Spherical Hankel function of 2nd kind.

    Defined as http://dlmf.nist.gov/10.47.E6,

    .. math::

        \hankel{2}{n}{z} = \sqrt{\frac{\pi}{2z}}
        \Hankel{2}{n + \frac{1}{2}}{z},

    where :math:`\Hankel{2}{n}{\cdot}` is the Hankel function of the
    second kind and n-th order, and :math:`z` its complex argument.

    Parameters
    ----------
    n : array_like
        Order of the spherical Hankel function (n >= 0).
    z : array_like
        Argument of the spherical Hankel function.

    """
    return spherical_jn(n, z) - 1j * spherical_yn(n, z)


print("plane wave")
# since the simulation works in normalized kr-domain, we only need to play
# with the far factor
far  = 200
# and the plane wave propagating direction
phi_pw = -1*np.pi/4



kr0 = far # or far*np.pi, very large for valid far/hf approx
M = 5*int(np.ceil(kr0))  # number of modes
L = 4*M  # number of secondary sources, >=M*2 to avoid spatial aliasing
wavelength = 1  # in m, value can be set arbitrarily
c = 343  # speed of sound in m/s, value can be set arbitrarily
k = 2*np.pi/wavelength  # in rad/m
f =  c/wavelength  # frequency in Hz
omega = 2*np.pi*f  # rad/s
r0 = kr0/k  # radius of spherical/circular array in m
phi_0 = np.arange(0,L)*2*np.pi/L
x_0 = r0*np.cos(phi_0)
y_0 = r0*np.sin(phi_0)
n_0 = [-x_0/r0, -y_0/r0]
k_pw = [np.cos(phi_pw), np.sin(phi_pw)]
w = (np.dot(k_pw, n_0)>=0)*1  # sec src sel


x_ref = r0 / 2
ma = np.arange(-M,M+1)
D_WFS = np.zeros((1, L),np.complex128)
D_WFS_FS_numeric = np.zeros((1, 2*M+1),np.complex128)
D_HOA_FS_analytic = np.zeros((1, 2*M+1),np.complex128)
D_WFS_partJ = np.zeros((1, L),np.complex128)
D_WFS_partSinc = np.zeros((1, L),np.complex128)
D_WFS_FS_J_numeric = np.zeros((1, 2*M+1),np.complex128)
D_WFS_FS_J_analytic  = np.zeros((1, 2*M+1),np.complex128)
D_WFS_FS_Sinc_numeric = np.zeros((1, 2*M+1),np.complex128)
D_WFS_FS_Sinc_analytic = np.zeros((1, 2*M+1),np.complex128)



D_WFS_partSinc[0, :] = w
D_WFS_partJ[0, :] =  - np.sqrt(1j*k*x_ref) * np.cos(phi_0-phi_pw) * \
                     np.exp(-1j*k*r0 * np.cos(phi_0-phi_pw))
# normalize to obtain 0dB in FS domain
# check in comparison to point source and here
D_WFS_partJ[0, :] =  1 / np.sqrt(2) * D_WFS_partJ[0, :]  # !?!?

D_WFS[0, :] = np.sqrt(8*np.pi) * D_WFS_partSinc[0, :] * D_WFS_partJ[0, :]



for m in ma:
    D_WFS_FS_numeric[0, m+M] = \
        1/(2*np.pi) * np.sum(D_WFS * np.exp(-1j*m*phi_0), 1) * 2*np.pi/L

    D_HOA_FS_analytic[0, m+M] = 0.5 * \
        2*1j/kr0 * (-1j)**np.abs(m) / spherical_hn2(np.abs(m), kr0) * \
        np.exp(-1j*m*phi_pw)

    D_WFS_FS_J_numeric[0, m+M] = \
        1/(2*np.pi) * np.sum(D_WFS_partJ * np.exp(-1j*m*phi_0), 1) * 2*np.pi/L

    D_WFS_FS_Sinc_numeric[0, m+M] = \
        1/(2*np.pi) * np.sum(D_WFS_partSinc * np.exp(-1j*m*phi_0), 1) * 2*np.pi/L

    D_WFS_FS_J_analytic[0, m+M] = - np.sqrt(1j*k*x_ref) * \
        np.exp(-1j*m*phi_pw) / (2*1j**(m-1)) * (jv(m-1,kr0) - jv(m+1,kr0))

    if m==0:
        D_WFS_FS_Sinc_analytic[0, m + M] = np.pi / 2 / np.pi
    else:
        D_WFS_FS_Sinc_analytic[0, m + M] =\
            -1j *\
            (np.exp(-1j * m * (phi_pw + np.pi / 2)) -
             np.exp(-1j * m * (phi_pw + 3 * np.pi / 2))) \
            / m / 2 / np.pi

# normalize to obtain 0dB in FS domain
# check in comparison to point source and here
D_WFS_FS_J_analytic[0,:] = 1 / np.sqrt(2) * D_WFS_FS_J_analytic[0,:]  # !?!?




ConvFS_Numeric = np.sqrt(8*np.pi) * \
    np.convolve(D_WFS_FS_J_numeric[0,:], D_WFS_FS_Sinc_numeric[0,:], mode="same")
ConvFS_Numeric = np.sqrt(8*np.pi) * \
    np.convolve(D_WFS_FS_J_analytic[0,:], D_WFS_FS_Sinc_analytic[0,:], mode="same")


m = np.arange(-M,M+1) / np.ceil(kr0)  # we plot over normalized m

fig, (ax1, ax2, ax3) = plt.subplots(3, 2)
fig.set_figheight(10)
fig.set_figwidth(10)


ax2[0].plot(m,D_HOA_FS_analytic[0,:].real,
            label="HOA analytic", color="C0")
ax2[0].plot(m,D_WFS_FS_numeric[0,:].real,
            label="WFS numeric", color="C1")
ax2[0].plot(m,ConvFS_Numeric.real, ":", color="C3")
ax2[0].set_xlim(-3, +3)
# ax2[0].set_xlabel(r"$m \, / \, \lceil k r_0 \rceil$, $m \in \mathrm{Z}$")
ax2[0].set_ylabel(r"Real(D(m))")
ax2[0].legend(loc="lower left")
ax2[0].grid(True)

ax2[1].plot(m,D_HOA_FS_analytic[0,:].imag,
            label="HOA analytic", color="C0")
ax2[1].plot(m,D_WFS_FS_numeric[0,:].imag,
            label="WFS numeric", color="C1")
ax2[1].plot(m,ConvFS_Numeric.imag, ":", color="C3")
ax2[1].set_xlim(-3, +3)
# ax2[1].set_xlabel(r"$m \, / \, \lceil k r_0 \rceil$, $m \in \mathrm{Z}$")
ax2[1].set_ylabel(r"Imag(D(m))")
ax2[1].legend(loc="lower left")
ax2[1].grid(True)


ax3[0].plot(m,np.abs(D_HOA_FS_analytic[0,:]),
            label="HOA analytic", color="C0")
ax3[0].plot(m,np.abs(D_WFS_FS_numeric[0,:]),
            label="WFS numeric", color="C1")
ax3[0].plot(m,np.abs(ConvFS_Numeric), ":", color="C3")
ax3[0].set_xlim(-3, +3)
ax3[0].set_xlabel(r"$m \, / \, \lceil k r_0 \rceil$, $m \in \mathrm{Z}$")
ax3[0].set_ylabel(r"|D(m)|")
ax3[0].legend(loc="lower left")
ax3[0].grid(True)

ax3[1].plot(m,20*np.log10(np.abs(D_HOA_FS_analytic[0,:])),
            label="HOA analytic", color="C0")
ax3[1].plot(m,20*np.log10(np.abs(D_WFS_FS_numeric[0,:])),
            label="WFS numeric", color="C1")
ax3[1].plot(m,20*np.log10(np.abs(ConvFS_Numeric)), ":", color="C3")
ax3[1].set_xlim(-3, +3)
ax3[1].set_ylim(-100, 20)
ax3[1].set_xlabel(r"$m \, / \, \lceil k r_0 \rceil$, $m \in \mathrm{Z}$")
ax3[1].set_ylabel(r"20 lg |D(m)| / dB")
ax3[1].legend(loc="lower left")
ax3[1].grid(True)
ax3[1].text(-2.75, 10 , "evanescent")
ax3[1].text(+1.25, 10 , "evanescent")
ax3[1].text(-2.8, -10, ''.join(['phiPW = ', str(phi_pw*180/np.pi), ' deg']))
ax3[1].text(-2.8, -30, ''.join(['kr0 = ', str(kr0)]))
rect_evanl = patches.Rectangle((-3,-100),2,120,linewidth=1,edgecolor='gray',
                         facecolor='gray', alpha=0.2)
rect_evanr = patches.Rectangle((1,-100),2,120,linewidth=1,edgecolor='gray',
                         facecolor='gray', alpha=0.2)
ax3[1].add_patch(rect_evanl)
ax3[1].add_patch(rect_evanr)


ax1[0].plot(m,0+D_WFS_FS_J_numeric[0,:].real,
            color="C0", label=(r"Re(D(m))/ ($8\pi$)"))
ax1[0].plot(m,D_WFS_FS_Sinc_numeric[0,:].real,
            color="C1", label=(r"Re(window(m))"))
ax1[0].plot(m,0+D_WFS_FS_J_analytic[0,:].real,":", color="C2", )
ax1[0].plot(m,D_WFS_FS_Sinc_analytic[0,:].real,":", color="C3", )
ax1[0].set_xlim(-3, +3)
# ax1[0].set_xlabel(r"$m \, / \, \lceil k r_0 \rceil$, $m \in \mathrm{Z}$")
ax1[0].set_ylabel(r"D$_{WFS}$(m)")
ax1[0].legend(loc="best")
ax1[0].grid(True)

ax1[1].plot(m,0+D_WFS_FS_J_numeric[0,:].imag,
            color="C0", label=(r"Im(D(m))/ ($8\pi$)"))
ax1[1].plot(m,D_WFS_FS_Sinc_numeric[0,:].imag,
            color="C1", label=(r"Im(window(m))"))
ax1[1].plot(m,0+D_WFS_FS_J_analytic[0,:].imag,":", color="C2")
ax1[1].plot(m,D_WFS_FS_Sinc_analytic[0,:].imag,":", color="C3")
ax1[1].set_xlim(-3, +3)
# ax1[1].set_xlabel(r"$m \, / \, \lceil k r_0 \rceil$, $m \in \mathrm{Z}$")
ax1[1].set_ylabel(r"D$_{WFS}$(m)")
ax1[1].legend(loc="best")
ax1[1].grid(True)


#plt.show()
plt.savefig('nfc_hoa_vs_WFS_drivingfunctions_plot_PW.pdf',
            dpi=300, bbox_inches='tight')




fig, ax = plt.subplots(1, 1)
fig.set_figheight(3.33)
fig.set_figwidth(5)

ax.plot(m,20*np.log10(np.abs(D_HOA_FS_analytic[0,:])),
            label="HOA analytic", color="C0")
ax.plot(m,20*np.log10(np.abs(D_WFS_FS_numeric[0,:])),
            label="WFS numeric", color="C1")
ax.plot(m,20*np.log10(np.abs(ConvFS_Numeric)), ":", color="C3")
ax.set_xlim(-3, +3)
ax.set_ylim(-100, 20)
ax.set_xlabel(r"$m \, / \, \lceil k r_0 \rceil$, $m \in \mathrm{Z}$")
ax.set_ylabel(r"20 lg |D(m)| / dB")
ax.legend(loc="lower left")
ax.grid(True)
ax.text(-2.75, 10 , "evanescent")
ax.text(+1.25, 10 , "evanescent")
ax.text(-2.8, -10, ''.join(['phiPW = ', str(phi_pw*180/np.pi), ' deg']))
ax.text(-2.8, -30, ''.join(['kr0 = ', str(kr0)]))
rect_evanl = patches.Rectangle((-3,-100),2,120,linewidth=1,edgecolor='gray',
                         facecolor='gray', alpha=0.2)
rect_evanr = patches.Rectangle((1,-100),2,120,linewidth=1,edgecolor='gray',
                         facecolor='gray', alpha=0.2)
ax.add_patch(rect_evanl)
ax.add_patch(rect_evanr)

plt.savefig('nfc_hoa_vs_WFS_drivingfunctions_plot_PW_single_dB.pdf',
            dpi=300, bbox_inches='tight')
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import jv, spherical_jn, spherical_yn
#import sfs



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



# since the simulation works in normalized kr-domain, we only need to play
# with the far factor
far  = 10
# and the plane wave propagating direction
phi_pw = 4*np.pi/4



kr0 = far # or far*np.pi, very large for valid far/hf approx
M = 5*int(np.ceil(kr0))  # number of modes
L = 4*M  # number of secondary sources, >=M*2 to avoid spatial aliasing
wavelength = 1  # in m, value can be set arbitrarily
c = 343  # speed of sound in m/s, value can be set arbitrarily
k = 2*np.pi/wavelength  # in rad/m
f =  c/wavelength  # frequency in Hz
omega = 2*np.pi*f  # rad/s
r0 = kr0/k  # radius of spherical/circular array in m
phi_start = phi_pw + np.pi/2  # don't change unless you're sure
phi_0 = phi_start + np.arange(0,L)*2*np.pi/L
x_0 = r0*np.cos(phi_0)
y_0 = r0*np.sin(phi_0)
phi_int = np.arange(0,L)*2*np.pi/L
idxpi = np.where(phi_int-np.pi ==0)
x_ref = r0 / 2
ma = np.arange(-M,M+1)
D_WFS = np.zeros((1, L),np.complex128)
D_WFS_FS_numeric = np.zeros((1, 2*M+1),np.complex128)
D_HOA_FS_analytic = np.zeros((1, 2*M+1),np.complex128)
D_WFS_partJ = np.zeros((1, L),np.complex128)
D_WFS_partSinc = np.ones((1, L),np.complex128)
D_WFS_FS_J_numeric = np.zeros((1, 2*M+1),np.complex128)
D_WFS_FS_J_analytic  = np.zeros((1, 2*M+1),np.complex128)
D_WFS_FS_Sinc_numeric = np.zeros((1, 2*M+1),np.complex128)
D_WFS_FS_Sinc_analytic = np.zeros((1, 2*M+1),np.complex128)



D_WFS[0, :] = - np.sqrt(8*np.pi*1j*k*x_ref) * np.cos(phi_0-phi_pw) * \
        np.exp(-1j*k*r0 * np.cos(phi_0-phi_pw))
D_WFS_partJ[0, :] = \
    np.cos(phi_0-phi_pw) * np.exp(-1j*k*r0 * np.cos(phi_0-phi_pw))
D_WFS[0, int(L/2):] = 0  # sec src sel = spatial window
D_WFS_partSinc[0, int(L/2):] = 0  # truncation window

for m in ma:
    D_WFS_FS_numeric[0, m+M] = \
        1/(2*np.pi) * np.sum(D_WFS * np.exp(-1j*m*phi_0), 1) * 2*np.pi/L
    D_HOA_FS_analytic[0, m+M] = \
        2*1j/kr0 * (-1j)**np.abs(m) / spherical_hn2(np.abs(m), kr0) * \
        np.exp(-1j*m*phi_pw)
# normalize
# TBD check how we get this offset between both approaches:
D_HOA_FS_analytic = D_HOA_FS_analytic / np.sqrt(2)

for m in ma:
    D_WFS_FS_J_numeric[0, m+M] = \
        1/(2*np.pi) * np.sum(D_WFS_partJ * np.exp(-1j*m*phi_0), 1) * 2*np.pi/L

    D_WFS_FS_Sinc_numeric[0, m+M] = \
        1/(2*np.pi) * np.sum(D_WFS_partSinc * np.exp(-1j*m*phi_0), 1) * 2*np.pi/L

    D_WFS_FS_J_analytic[0, m+M] = \
        np.exp(-1j*m*phi_pw) / (2*1j**(m-1)) * (jv(m-1,kr0) - jv(m+1,kr0))

    if m==0:
        D_WFS_FS_Sinc_analytic[0, m + M] = np.pi / 2 / np.pi
    else:
        D_WFS_FS_Sinc_analytic[0, m + M] =\
            -1j *\
            (np.exp(-1j * m * (phi_pw + np.pi / 2)) -
             np.exp(-1j * m * (phi_pw + 3 * np.pi / 2))) \
            / m / 2 / np.pi

ConvFS_Numeric = - np.sqrt(8*np.pi*1j*k*x_ref) * \
                 np.convolve(D_WFS_FS_J_analytic[0,:],
                             D_WFS_FS_Sinc_analytic[0,:], mode="same")

m = np.arange(-M,M+1) / np.ceil(kr0)

fig, (ax1, ax2, ax3) = plt.subplots(3, 2)
fig.set_figheight(10)
fig.set_figwidth(10)

ax1[0].plot(m,D_HOA_FS_analytic[0,:].real, label="HOA analytic", color="C0")
ax1[0].plot(m,D_WFS_FS_numeric[0,:].real, label="WFS numeric", color="C1")
ax1[0].plot(m,ConvFS_Numeric.real, ":", label="WFS numeric conv", color="C3")
ax1[0].set_xlim(-3, +3)
ax1[0].set_xlabel(r"$m \, / \, \lceil k r_0 \rceil$, $m \in \mathrm{Z}$")
ax1[0].set_ylabel(r"Real(D(m))")
ax1[0].legend(loc="lower left")
ax1[0].grid(True)

ax1[1].plot(m,D_HOA_FS_analytic[0,:].imag, label="HOA analytic", color="C0")
ax1[1].plot(m,D_WFS_FS_numeric[0,:].imag, label="WFS numeric", color="C1")
ax1[1].plot(m,ConvFS_Numeric.imag, ":", label="WFS numeric conv", color="C3")
ax1[1].set_xlim(-3, +3)
ax1[1].set_xlabel(r"$m \, / \, \lceil k r_0 \rceil$, $m \in \mathrm{Z}$")
ax1[1].set_ylabel(r"Imag(D(m))")
ax1[1].legend(loc="lower left")
ax1[1].grid(True)



ax2[0].plot(m,np.abs(D_HOA_FS_analytic[0,:]), label="HOA analytic", color="C0")
ax2[0].plot(m,np.abs(D_WFS_FS_numeric[0,:]), label="WFS numeric", color="C1")
ax2[0].plot(m,np.abs(ConvFS_Numeric), ":", label="WFS numeric conv",
            color="C3")
ax2[0].set_xlim(-3, +3)
ax2[0].set_xlabel(r"$m \, / \, \lceil k r_0 \rceil$, $m \in \mathrm{Z}$")
ax2[0].set_ylabel(r"|D(m)|")
ax2[0].legend(loc="lower left")
ax2[0].grid(True)

ax2[1].plot(m,20*np.log10(np.abs(D_HOA_FS_analytic[0,:])), label="HOA "
                                                                 "analytic",
            color="C0")
ax2[1].plot(m,20*np.log10(np.abs(D_WFS_FS_numeric[0,:])), label="WFS "
                                                                "numeric",
            color="C1")
ax2[1].plot(m,20*np.log10(np.abs(ConvFS_Numeric)), ":", label="WFS numeric "
                                                              "conv",
            color="C3")
ax2[1].set_xlim(-3, +3)
ax2[1].set_ylim(-100, 20)
ax2[1].set_xlabel(r"$m \, / \, \lceil k r_0 \rceil$, $m \in \mathrm{Z}$")
ax2[1].set_ylabel(r"20 lg |D(m)| / dB")
ax2[1].legend(loc="best")
ax2[1].grid(True)



ax3[0].plot(m,D_WFS_FS_J_numeric[0,:].real, label=(r"Re($J_{m-1}-J_{m+1}$)"))
ax3[0].plot(m,1+D_WFS_FS_Sinc_numeric[0,:].real, label=(r"Re(Sinc)+1"))
ax3[0].plot(m,D_WFS_FS_J_analytic[0,:].real,":")
ax3[0].plot(m,1+D_WFS_FS_Sinc_analytic[0,:].real,":")
ax3[0].set_xlim(-3, +3)
ax3[0].set_xlabel(r"$m \, / \, \lceil k r_0 \rceil$, $m \in \mathrm{Z}$")
ax3[0].set_ylabel(r"D$_{WFS}$(m)")
ax3[0].legend(loc="best")
ax3[0].grid(True)

ax3[1].plot(m,D_WFS_FS_J_numeric[0,:].imag, label=(r"Imag($J_{m-1}-J_{m+1}$)"))
ax3[1].plot(m,1+D_WFS_FS_Sinc_numeric[0,:].imag, label=(r"Imag(Sinc)+1"))
ax3[1].plot(m,D_WFS_FS_J_analytic[0,:].imag,":")
ax3[1].plot(m,1+D_WFS_FS_Sinc_analytic[0,:].imag,":")
ax3[1].set_xlim(-3, +3)
ax3[1].set_xlabel(r"$m \, / \, \lceil k r_0 \rceil$, $m \in \mathrm{Z}$")
ax3[1].set_ylabel(r"D$_{WFS}$(m)")
ax3[1].legend(loc="best")
ax3[1].grid(True)

#plt.show()
plt.savefig('nfc_hoa_vs_WFS_drivingfunctions_plot.pdf', dpi=72,
            bbox_inches='tight')





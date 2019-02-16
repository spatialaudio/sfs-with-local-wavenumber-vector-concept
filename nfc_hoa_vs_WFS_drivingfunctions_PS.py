# NFC-HOA vs. WFS driving POINT SOURCE functions equivalence?!
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


print("point source")
# since the simulation works in normalized kr-domain, we only need to play
# with the far factor

#TBD: check this prefactors and the analytic solution of Sinc and Jm-1 / Jm+1
#it seems to be not completely wrong, however we have a missing dependency
#of krs and kr0, sign mismatch and normalize mismatch
#re-check with AES140th Hahn et. al
far_r0 = 20
normfac = 0.86

far_r0 = 10
normfac = 0.6283

far_r0 = 2
normfac = 0.29

far_r0 = 1
normfac = 0.2

far_r0 = 100
normfac = 2

far_r0 = 200
normfac = 2.6

far_r0 = 31.4159
normfac = 1


far_r0 = 20
normfac = 0.86

far_ps = 1000*np.pi
# and the point source position angle
phi_ps = 4*np.pi/4



kr0 = far_r0 # or far*np.pi, very large for valid far/hf approx
krs = far_ps
M = 5*int(np.ceil(kr0))  # number of modes
L = 4*M  # number of secondary sources, >=M*2 to avoid spatial aliasing
wavelength = 1  # in m, value can be set arbitrarily
c = 343  # speed of sound in m/s, value can be set arbitrarily
k = 2*np.pi/wavelength  # in rad/m
f =  c/wavelength  # frequency in Hz
omega = 2*np.pi*f  # rad/s
r0 = kr0/k  # radius of spherical/circular array in m
phi_0 = np.arange(0,L)*2*np.pi/L
x0 = np.array([r0*np.cos(phi_0), r0*np.sin(phi_0)])
n_0 = -x0/r0
x_ref = np.zeros((2,1))
x_ref[:,0] = 0


rs = krs/k
Aps = 4*np.pi*rs
x_ps = np.zeros((2,1))
x_ps[0,0] = np.cos(phi_ps)*rs
x_ps[1,0] = np.sin(phi_ps)*rs
k_ps = ( x0 - x_ps ) / np.linalg.norm(x0-x_ps, axis=0)
xoxref = np.linalg.norm(x0 - x_ref, axis=0)
xoxps = np.linalg.norm(x0 - x_ps, axis=0)
w =  ((k_ps * n_0).sum(axis=0)>=0)*1  # sec src sel

print(w)

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
ConvFS_Numeric = np.zeros((1, 2*M+1),np.complex128)

D_WFS[0, :] = w * 8*np.pi*np.sqrt(1j*k) * \
               np.sqrt((xoxref*xoxps)/(xoxref+xoxps)) * \
               (k_ps * n_0).sum(axis=0) * np.exp(-1j*k*xoxps) / \
               (4*np.pi*xoxps) * Aps
D_WFS_partJ[0, :] = np.sqrt((xoxref*xoxps)/(xoxref+xoxps)) * \
               (k_ps * n_0).sum(axis=0) * np.exp(-1j*k*xoxps) / \
               (4*np.pi*xoxps) * Aps
D_WFS_partSinc[0, :] = w

for m in ma:
    D_WFS_FS_numeric[0, m+M] = \
        1/(2*np.pi) * np.sum(D_WFS * np.exp(-1j*m*phi_0), 1) * 2*np.pi/L
    D_HOA_FS_analytic[0, m+M] = Aps * 2/r0 * \
                                spherical_hn2(np.abs(m), krs) / \
                                spherical_hn2(np.abs(m), kr0) *\
    np.exp(-1j*m*phi_ps)

# normalize
# TBD check how we get this offset between both approaches:
D_HOA_FS_analytic = D_HOA_FS_analytic / np.sqrt(2*np.pi)
# also check the overall +20dB level in FS domain

phi0_active_idx = np.where(w)[0]  # get active sources index

print(phi_0[phi0_active_idx[0]]*180/np.pi, phi_0[phi0_active_idx[
    -1]]*180/np.pi)

for m in ma:
    D_WFS_FS_J_numeric[0, m+M] = \
        1/(2*np.pi) * np.sum(D_WFS_partJ * np.exp(-1j*m*phi_0), 1) * \
        2*np.pi/L

    D_WFS_FS_Sinc_numeric[0, m+M] = \
        1/(2*np.pi) * np.sum(D_WFS_partSinc * np.exp(-1j*m*phi_0), 1) * \
        2*np.pi/L

    D_WFS_FS_J_analytic[0, m+M] = \
        1j*k*spherical_hn2(np.abs(m), krs) * np.exp(-1j*m*phi_ps) / \
        (4*np.pi*1j**(m-np.abs(m))) * \
        (jv(m-1,kr0) - jv(m+1,kr0)) * Aps * 1/1j**(-1) * normfac

    if m == 0:
        D_WFS_FS_Sinc_analytic[0, m + M] = np.pi / 2 / np.pi
    else:
        tmp = np.argmax(w)  # this needs double check and analytic derivation
        # also this works only if active src are 'embedded' within nonactive
        # which is for pi/2 < phi_ps < 3pi/2
        D_WFS_FS_Sinc_analytic[0, m + M] =1/(-1j*m) * 1/(2*np.pi) \
                                          * (np.exp(-1j*m*phi_0[
            phi0_active_idx[-1]])-np.exp(-1j*m*phi_0[phi0_active_idx[0]]))

#ConvFS_Numeric = - np.sqrt(8*np.pi*1j*k*x_ref) * \
#                 np.convolve(D_WFS_FS_J_analytic[0,:],
#                             D_WFS_FS_Sinc_analytic[0,:], mode="same")

ConvFS_Numeric = 8*np.pi*np.sqrt(1j*k) * np.convolve(D_WFS_FS_J_numeric[0,:],
                             D_WFS_FS_Sinc_numeric[0,:], mode="same")


m = np.arange(-M,M+1) / np.ceil(kr0)  # we plot over normalized m
norm_val = np.max(np.abs(D_HOA_FS_analytic[0,:]))

fig, (ax1, ax2, ax3) = plt.subplots(3, 2)
fig.set_figheight(10)
fig.set_figwidth(10)


ax1[0].plot(m,D_HOA_FS_analytic[0,:].real/norm_val,
            label="HOA analytic", color="C0")
ax1[0].plot(m,D_WFS_FS_numeric[0,:].real/norm_val,
            label="WFS numeric", color="C1")
ax1[0].plot(m,ConvFS_Numeric.real/norm_val, ":", color="C3")
ax1[0].set_xlim(-3, +3)
# ax1[0].set_xlabel(r"$m \, / \, \lceil k r_0 \rceil$, $m \in \mathrm{Z}$")
ax1[0].set_ylabel(r"Real(D(m)) / max |D$_{HOA}$(m)|")
ax1[0].legend(loc="lower left")
ax1[0].grid(True)

ax1[1].plot(m,D_HOA_FS_analytic[0,:].imag/norm_val,
            label="HOA analytic", color="C0")
ax1[1].plot(m,D_WFS_FS_numeric[0,:].imag/norm_val,
            label="WFS numeric", color="C1")
ax1[1].plot(m,ConvFS_Numeric.imag/norm_val, ":", color="C3")
ax1[1].set_xlim(-3, +3)
# ax1[1].set_xlabel(r"$m \, / \, \lceil k r_0 \rceil$, $m \in \mathrm{Z}$")
ax1[1].set_ylabel(r"Imag(D(m))  / max |D$_{HOA}$(m)|")
ax1[1].legend(loc="lower left")
ax1[1].grid(True)


ax2[0].plot(m,np.abs(D_HOA_FS_analytic[0,:]/norm_val),
            label="HOA analytic", color="C0")
ax2[0].plot(m,np.abs(D_WFS_FS_numeric[0,:]/norm_val),
            label="WFS numeric", color="C1")
ax2[0].plot(m,np.abs(ConvFS_Numeric/norm_val), ":", color="C3")
ax2[0].set_xlim(-3, +3)
# ax2[0].set_xlabel(r"$m \, / \, \lceil k r_0 \rceil$, $m \in \mathrm{Z}$")
ax2[0].set_ylabel(r"|D(m)|  / max |D$_{HOA}$(m)|")
ax2[0].legend(loc="lower left")
ax2[0].grid(True)
ax2[0].text(-2.8, 0.9, ''.join(['phiPS=', str(phi_ps*180/np.pi), 'deg']))
ax2[0].text(-2.8, 0.8, ''.join(['krs=', str(krs)]))
ax2[0].text(-2.8, 0.7, ''.join(['kr0=', str(kr0)]))


ax2[1].plot(m,20*np.log10(np.abs(D_HOA_FS_analytic[0,:]/norm_val)),
            label="HOA analytic", color="C0")
ax2[1].plot(m,20*np.log10(np.abs(D_WFS_FS_numeric[0,:]/norm_val)),
            label="WFS numeric", color="C1")
ax2[1].plot(m,20*np.log10(np.abs(ConvFS_Numeric/norm_val)), ":", color="C3")
ax2[1].set_xlim(-3, +3)
ax2[1].set_ylim(-100, 20)
# ax2[1].set_xlabel(r"$m \, / \, \lceil k r_0 \rceil$, $m \in \mathrm{Z}$")
ax2[1].set_ylabel(r"20 lg |D(m)| / dB$_{\mathrm{rel}}$")
ax2[1].legend(loc="lower left")
ax2[1].grid(True)
ax2[1].text(-2.75, 10 , "evanescent")
ax2[1].text(+1.25, 10 , "evanescent")
rect_evanl = patches.Rectangle((-3,-100),2,120,linewidth=1,edgecolor='gray',
                         facecolor='gray', alpha=0.2)
rect_evanr = patches.Rectangle((1,-100),2,120,linewidth=1,edgecolor='gray',
                         facecolor='gray', alpha=0.2)
ax2[1].add_patch(rect_evanl)
ax2[1].add_patch(rect_evanr)


ax3[0].plot(m,0.1+D_WFS_FS_J_numeric[0,:].real/norm_val,
            color="C0", label=(r"Re($J_{m-1}-J_{m+1}$), off 0.1"))
ax3[0].plot(m,D_WFS_FS_Sinc_numeric[0,:].real/norm_val,
            color="C1", label=(r"Re(Sinc)"))
ax3[0].plot(m,0.1+D_WFS_FS_J_analytic[0,:].real/norm_val,":", color="C2", )
ax3[0].plot(m,D_WFS_FS_Sinc_analytic[0,:].real/norm_val,":", color="C3", )
ax3[0].set_xlim(-3, +3)
ax3[0].set_xlabel(r"$m \, / \, \lceil k r_0 \rceil$, $m \in \mathrm{Z}$")
ax3[0].set_ylabel(r"D$_{WFS}$(m) / max |D$_{HOA}$(m)|")
ax3[0].legend(loc="best")
ax3[0].grid(True)

ax3[1].plot(m,0.1+D_WFS_FS_J_numeric[0,:].imag/norm_val,
            color="C0", label=(r"Imag($J_{m-1}-J_{m+1}$), off 0.1"))
ax3[1].plot(m,D_WFS_FS_Sinc_numeric[0,:].imag/norm_val,
            color="C1", label=(r"Imag(Sinc)"))
ax3[1].plot(m,0.1+D_WFS_FS_J_analytic[0,:].imag/norm_val,":", color="C2")
ax3[1].plot(m,D_WFS_FS_Sinc_analytic[0,:].imag/norm_val,":", color="C3")
ax3[1].set_xlim(-3, +3)
ax3[1].set_xlabel(r"$m \, / \, \lceil k r_0 \rceil$, $m \in \mathrm{Z}$")
ax3[1].set_ylabel(r"D$_{WFS}$(m) / max |D$_{HOA}$(m)|")
ax3[1].legend(loc="best")
ax3[1].grid(True)


#plt.show()
plt.savefig('nfc_hoa_vs_WFS_drivingfunctions_plot_PS.pdf',
            dpi=300, bbox_inches='tight')

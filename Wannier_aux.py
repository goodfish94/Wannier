import numpy as np


def get_pauli_mat():
    s0 = np.asarray([[1.,0.],[0.,1.]], dtype=np.complex128)
    s1 = np.asarray([[0., 1.], [1., 0.]], dtype=np.complex128)
    s2 = np.asarray([[0., -1.j], [1.j, 0.]], dtype=np.complex128)
    s3 = np.asarray([[1., 0.], [0., -1.]], dtype=np.complex128)
    return [s0,s1,s2,s3]


def generate_kpath(HSP_list, Nk=20):
    # for given kpt_list=[k0, k1, ..., k0], generate kpoints on the path.
    # each path has Nk+1 pts, from ki to ki+1
    kpath = []
    for k1, k2 in zip(HSP_list[0:-1], HSP_list[1:]):
        kstep = (k2 - k1) / Nk
        for ik in range(Nk + 1):
            kpt = k1 + ik * kstep
            kpath.append(kpt)
    return np.asarray( kpath )



def generate_kmesh_3d(b123, Nk=20):
    # for given kpt_list=[k0, k1, ..., k0], generate kpoints on the path.
    # each path has Nk+1 pts, from ki to ki+1
    kmesh = [] 
    for i in range(0, Nk):
        for j in range(0, Nk):
            for k in range(0, Nk):
                kpt = (i / Nk) * b123[0] + (j / Nk) * b123[1] + (k / Nk) * b123[2]
                kmesh.append(kpt)
    kmesh = np.asarray(kmesh)
    return kmesh 


def gen_lenk(klist):
    # for given kpt_list=[k0, k1, ..., k0], generate kpoints on the path.
    # each path has Nk+1 pts, from ki to ki+1

    len_it = 0
    lenk = []
    for i in range(0, len(klist) - 1):
        lenk.append(len_it)
        len_it += np.linalg.norm(klist[i + 1] - klist[i])
    lenk.append(len_it)

    return np.asarray(lenk)


def read_hrdat(filename):
    # return hrdat of shape (2*max_R+1, 2*max_R+1, 2*max_R+1, num_wann, num_wann)
    # [i, j, k, n1, n2] is the hopping of R=(i,j,k) between n1, n2 orbits

    max_R = 0
    with open(filename, 'r') as f:
        f = f.readlines()
        for cnt, line in enumerate(f):
            l = line.strip().split()
            if len(l) == 7 and len(l[-1]) > 2:
                idx = [int(i) for i in l[0:5]]
                max_R = max([max_R, np.abs(idx[0]), np.abs(idx[1]), np.abs(idx[2])])
    print('max R of hr=', max_R)

    with open(filename, 'r') as f:
        f = f.readlines()
        num_wann = int(f[1])

        hrdat = np.zeros((2 * max_R + 1, 2 * max_R + 1, 2 * max_R + 1, num_wann, num_wann), dtype=complex)

        for cnt, line in enumerate(f):
            l = line.strip().split()
            if len(l) == 7 and len(l[-1]) > 2:
                idx = [int(i) for i in l[0:5]]

                hopping = float(l[5]) + 1j * float(l[6])
                hrdat[idx[0], idx[1], idx[2], idx[3] - 1, idx[4] - 1] = hopping

    return hrdat


def get_hk_from_hop(k, Rlist, hop, rsub):
    norb = hop.shape[-1]

    hk = np.zeros((len(k), hop.shape[-1], hop.shape[-2]), dtype=np.complex128)
    for i in range(0, len(Rlist)):
        for io1 in range(0, norb):
            for io2 in range(0, norb):
                r = Rlist[i] + rsub[io2] - rsub[io1]
                hk[:, io1, io2] += hop[i, io1, io2] * np.exp(1j * k @ r)
    return hk



def get_dhk_from_hop(k, Rlist, hop, rsub ):
    norb = hop.shape[-1]

    # hk = np.zeros((len(k), hop.shape[-1], hop.shape[-2]), dtype=np.complex128)
    dhk =  np.zeros((k.shape[-1], len(k), hop.shape[-1], hop.shape[-2]), dtype=np.complex128)
    for i in range(0, len(Rlist)):
        for io1 in range(0, norb):
            for io2 in range(0, norb):
                r = Rlist[i] + rsub[io2] - rsub[io1]
                dhk[:, :, io1, io2] += np.einsum('k, d->dk', hop[i,io1, io2] *  np.exp(1j * k @ r),  1j * r )
    return dhk


def gen_hk(k, hrdat, rsub):
    norb = hrdat.shape[-1]
    hk = np.zeros((len(k), norb, norb), dtype=np.complex128)
    R = hrdat.shape[0]
    Rmax = np.round( (R-1)/2 ).astype(np.int64)
    for ix in range(-Rmax,Rmax+1):
        for iy in range(-Rmax, Rmax + 1):
            for iz in range(-Rmax, Rmax + 1):
                for io1 in range(0,norb):
                    for io2 in range(0,norb):
                        dr = np.asarray([ ix,iy,iz]) + rsub[io2] - rsub[io1]
                        hk[:, io1,io2] += hrdat[ix,iy,iz, io1,io2] * np.exp( 1j * k @ dr)
    return hk


def avec_to_bvec(a123):
    a1, a2, a3 = a123[0], a123[1], a123[2]
    v = np.dot( a1, np.cross(a2,a3))

    b1 = 2.0 * np.pi / v * np.cross(a2,a3)
    b2 = 2.0 * np.pi / v * np.cross(a3, a1)
    b3 = 2.0 * np.pi / v * np.cross(a1, a2)

    return np.asarray([b1,b2,b3])




def avec_to_bvec_2D(a12):
    a1, a2, a3 = np.zeros(3), np.zeros(3), np.zeros(3)
    a1[0:2] = a12[0]
    a2[0:2] = a12[1]
    a3[2] = 1.0 
    
    v = np.dot( a1, np.cross(a2,a3))

    b1 = 2.0 * np.pi / v * np.cross(a2,a3)
    b2 = 2.0 * np.pi / v * np.cross(a3, a1)
    b3 = 2.0 * np.pi / v * np.cross(a1, a2)

    return np.asarray([b1[0:2], b2[0:2]]) 


def get_kbz_2d(N):
    k = np.zeros((N,N,2), dtype=np.complex128)
    for i in range(0,N):
        for j in range(0,N):
            k[i,j] = np.asarray([i/N, j/N])
    k = np.reshape(k, (N*N,2))
    return k



def get_hop_from_hk(Rlist, hk, klist):
    # version without sublattice 
    hop = np.zeros((len(Rlist)), dtype=np.complex128)
    
    for i, R in enumerate(Rlist):
        kR = klist@R 
        form = np.exp( 2.0 * np.pi * (-1j) * kR )
        hop[i] = np.sum( form * hk )/len(klist) 
        
    return hop  





def get_unitcell_boundary( a, ax, shift_vec = np.asarray([0., 0.])):

    
    cut = 0.5 * a / np.sqrt(3.0)
    ax.plot([ shift_vec[0]-cut,shift_vec[0]+ cut], [shift_vec[1]+0.5 * a, shift_vec[1]+0.5 * a], color='black')
    ax.plot([shift_vec[0]-cut, shift_vec[0]+cut], [shift_vec[1]-0.5 * a, shift_vec[1]-0.5 * a], color='black')

    x = np.linspace(0, 0.5 * a, 100)
    y = - x / np.sqrt(3.0) + a / np.sqrt(3)
    ax.plot(shift_vec[0]+y, shift_vec[1]+x, color='black')

    x = np.linspace(0, 0.5 * a, 100)
    y = x / np.sqrt(3.0) - a / np.sqrt(3)
    ax.plot(shift_vec[0]+y, shift_vec[1]+x, color='black')

    x = np.linspace(0, -0.5 * a, 100)
    y = x / np.sqrt(3.0) + a / np.sqrt(3)
    ax.plot(shift_vec[0]+y, shift_vec[1]+x, color='black')

    x = np.linspace(0, -0.5 * a, 100)
    y = -x / np.sqrt(3.0) - a / np.sqrt(3)
    ax.plot(shift_vec[0]+y, shift_vec[1]+x, color='black')






def if_in_hex_bz(k, b1,b2,b3):
    # check if vector in the hex bz 
    bmode = np.linalg.norm(b1)
    if( np.abs(k@b1) > 0.5 * bmode**2 ): return False 
    if( np.abs(k@b2) > 0.5 * bmode**2 ): return False 
    if( np.abs(k@b3) > 0.5 * bmode**2 ): return False 
    return True 



def get_unitcell_boundary_shift( a, Rshift, ax):

    cut = 0.5 * a / np.sqrt(3.0)
    ax.plot( np.asarray( [-cut, cut]  )+Rshift[0], np.asarray( [0.5 * a, 0.5 * a]) + Rshift[1], color='black')
    ax.plot( np.asarray( [-cut, cut] )+Rshift[0], np.asarray( [-0.5 * a, -0.5 * a]) + Rshift[1], color='black')

    x = np.linspace(0, 0.5 * a, 100)
    y = - x / np.sqrt(3.0) + a / np.sqrt(3)
    ax.plot(y +Rshift[0], x+ Rshift[1], color='black')

    x = np.linspace(0, 0.5 * a, 100)
    y = x / np.sqrt(3.0) - a / np.sqrt(3)
    ax.plot(y +Rshift[0], x+ Rshift[1], color='black')

    x = np.linspace(0, -0.5 * a, 100)
    y = x / np.sqrt(3.0) + a / np.sqrt(3)
    ax.plot(y +Rshift[0], x+ Rshift[1], color='black')

    x = np.linspace(0, -0.5 * a, 100)
    y = -x / np.sqrt(3.0) - a / np.sqrt(3)
    ax.plot(y +Rshift[0], x+ Rshift[1], color='black')







def get_bz_boundary( b , ax):

    a = b

    cut = 0.5 * a / np.sqrt(3.0)
    ax.plot( [0.5 * a, 0.5 * a], [-cut, cut], color='black')
    ax.plot( [-0.5 * a, -0.5 * a], [-cut, cut], color='black')

    x = np.linspace(0, 0.5 * a, 100)
    y = - x / np.sqrt(3.0) + a / np.sqrt(3)
    ax.plot(x, y, color='black')

    x = np.linspace(0, 0.5 * a, 100)
    y = x / np.sqrt(3.0) - a / np.sqrt(3)
    ax.plot(x, y, color='black')

    x = np.linspace(0, -0.5 * a, 100)
    y = x / np.sqrt(3.0) + a / np.sqrt(3)
    ax.plot(x, y, color='black')

    x = np.linspace(0, -0.5 * a, 100)
    y = -x / np.sqrt(3.0) - a / np.sqrt(3)
    ax.plot(x, y, color='black')

def read_poscar(filename='POSCAR'):
    with open(filename, 'r') as f:
        f = f.readlines()
        a = []
        alines = f[2:5]
        for line in alines:
            a.append([float(i) for i in line.split()])

        atom_label = [ia for ia in f[5].split()]
        atom_num = [int(ia) for ia in f[6].split()]

        pos = []
        assert 'direct' in f[7].lower()
        poslines = f[8:]
        for line in poslines:
            if len(line) < 20:
                continue
            pos.append([float(i) for i in line.split()])

    a123 = np.array(a)
    pos = np.array(pos)
    pos_cart = np.zeros_like(pos)
    for ith, p in enumerate(pos):
        pos_cart[ith] = p @ a123


    return a123, pos, atom_label, atom_num, pos_cart




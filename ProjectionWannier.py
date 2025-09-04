# Construct Wannier via projectio

import numpy as np


class ProjWan:
    def __init__(self):
        pass

    def import_wavefunction_and_ham(self, egv, hk ):
        """
        egv: shape = [nk, norb, nband]
        """
        self.egv = egv
        self.nk = len(egv)
        self.norb = egv.shape[1]
        self.nband = egv.shape[2]

        self.hk = hk




    def wannier_construction(self, psi):
        """
       trial wavefunction: psi  [nk, norb, ntrial]
       """
        if (psi.shape[0] != self.nk or psi.shape[1] != self.norb):
            raise ValueError('Shape of trial does not match band basis', psi.shape, self.nk, self.norb)
        self.psi = psi
        self.nwan = self.psi.shape[2]

        self.A = np.einsum('kQn, kQa->kna', np.conj(self.egv), self.psi)
        self.S = np.einsum('kna, knb->kab', np.conj(self.A), self.A )

        egS, egvS = np.linalg.eigh(self.S)
        min_S = np.min( egS)
        print('Minimum value of S eigen = ', min_S)

        sqrt_invS = np.einsum('kin, kjn, kn->kij', egvS, np.conj(egvS), 1.0/(np.sqrt(egS)))
        self.wannier = np.einsum('kna, kQn, kab->kQb', self.A, self.egv, sqrt_invS )
        return self.wannier

    def print_overlapping(self):

        for ib in range(0, self.nband ):
            tot_ov = 0.0
            for iw in range(self.wannier.shape[2]):
                overlap = np.sum(np.conj(self.egv[:, :, ib]) * self.wannier[:, :, iw], axis=1)
                overlap = np.mean(np.abs(overlap) ** 2)
                print('overlap btw band %d and wannier %d' % (ib, iw), overlap)
                tot_ov += overlap
            print('tot overlap with band %d = ' % ib, tot_ov)
            print('\n')

    def construct_wannier_band(self):
        hk_wan = np.einsum('kab, kan, kbm->knm', self.hk, np.conj(self.wannier), self.wannier)
        return hk_wan


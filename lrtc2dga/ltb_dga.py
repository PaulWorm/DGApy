#! /usr/bin/env python
import sys
import os
import logging
import argparse
from argparse import RawTextHelpFormatter

import numpy as np
from structure.inout import h5output
from structure.tb    import TightBinding
from structure.aux   import LogFormatter

'''
ltb: Pre-processing of tight-binding structures for the LRTC code
'''

__author__     = 'Matthias Pickem'
__maintainer__ = 'Matthias Pickem'
__email__      = 'matthias.pickem@gmail.com'
__version__    = '0.3'
__status__     = 'Dev'
__license__    = 'GPLv3'
__credits__    = ['Matthias Pickem', 'Emanuele Maggio', 'Jan M. Tomczak']

def parse_args(args=None):
  parser = argparse.ArgumentParser(
    description='Argument parser for the TB pre-processing for LRTC', \
    formatter_class=RawTextHelpFormatter,
    epilog='''
tb_file:
  begin hopping:      hopping parameter [eV]
                      sign convention e(k) ~ -t * (1-2.delta_(R,0).dleta_(l,l')) e^{ikR}
  begin atoms:        fractional atomic positions inside unit cell
  begin real_lattice: lattice vectors [Angstrom]
                      each row = 1 vector, each column = Cartesian x/y/z component


begin hopping
# x  y  z  orbital1 orbital2 hopping.real [hopping.imag]
  0 +1  0  1        1        0.25
  0 -1  0  1        1        0.25 0.005
...
end hopping

begin atoms
# ineq (starting at 1) x y z
  1                    0 0 0
...
end atoms

begin real_lattice
# x y z
  5 0 0
  0 5 0
  0 0 1
end real_lattice
''')

  # mandatory
  parser.add_argument('tb_file', type=str, help='tight binding file')
  parser.add_argument('nkx', type=int, help='number of k-points in x-direction')
  parser.add_argument('nky', type=int, help='number of k-points in y-direction')
  parser.add_argument('nkz', type=int, help='number of k-points in z-direction')
  parser.add_argument('filling', type=float, help='number of electrons in the system (1 filled band = 2 electrons)')

  # optional
  parser.add_argument('-o', '--output', default=None, help='Output file name')
  parser.add_argument('-p', '--plot', default=False, action='store_true', help='Plot bands / DOS / NOS')
  parser.add_argument('--kshift', default=False, help='shift k-grid by half a k-point', action='store_true')
  parser.add_argument('--mushift', default=False, help='shift energies such that the chemical potential is at mu = 0', action='store_true')
  parser.add_argument('--intraonly', default=False, help='only save intra-band elements', action='store_true')
  parser.add_argument('--intra', type=float, help='set all intra-band elements to given value')
  parser.add_argument('--inter', type=float, help='set all inter-band elements to given value')
  parser.add_argument('--red', default=False, help='make k-grid reducible', action='store_true')
  parser.add_argument('--mu', type=float, help='use provided chemical potential instead of provided filling (debugging purposes)')
  parser.add_argument('--debug', help=argparse.SUPPRESS, default=False, action='store_true')
  return parser.parse_args(args)

def main():
  error = lambda string: sys.exit('\nltb: {}'.format(string))
  args = parse_args()
  debug = args.debug

  ''' define logging '''
  logger = logging.getLogger()
  logger.setLevel(logging.DEBUG if debug else logging.INFO)
  console = logging.StreamHandler()
  console.setFormatter(LogFormatter())
  console.setLevel(logging.DEBUG if debug else logging.INFO)
  logger.addHandler(console)

  # create tightbinding object
  irr = not args.red
  try:
    tb = TightBinding(nkx=args.nkx, \
                      nky=args.nky, \
                      nkz=args.nkz, \
                      irreducible=irr, \
                      kshift=args.kshift)
  except:
    error(str(e)+'\nExit.')

  # compute the dispersion, etc and output it
  try:
    tb.computeData(tbfile=args.tb_file, \
                   charge=args.filling, \
                   mu=args.mu,
                   mushift=args.mushift)
  except Exception as e:
    error(str(e)+'\nExit.')

  if args.intra is not None:
    tb.setDiagonal(args.intra)
  if args.inter is not None:
    tb.setOffDiagonal(args.inter)

  if args.intraonly:
    tb.bopticfull = False
    tb.opticfull = False

  fname = args.output if args.output is not None else \
          'lrtc-tb-{}-{}-{}-{}.hdf5'.format(tb.nkx,tb.nky,tb.nkz,'irr' if irr else 'red')
  h5output(fname, tb, tb)

  logger.info("Output file {!r} successfully created.".format(fname))
  logger.info("File size: {} MB.".format(os.stat(fname).st_size/1000000))

  if args.plot:
    try:
      logging.getLogger("matplotlib").setLevel(logging.WARNING)
      import matplotlib.pyplot as plt
    except ImportError:
      error('Debug option requires matplotlib library')

    for iband in range(tb.energyBandMax):
      plt.plot(tb.energies[0][:,iband], label='band {}'.format(iband+1), lw=2)
      mean = np.mean(tb.energies[0][:,iband])
    plt.axhline(tb.mu, label='mu_tb = {:.3f}'.format(tb.mu), color='black', lw=1, ls='--')
    plt.xlabel(r'$k_i$')
    plt.ylabel(r'$\varepsilon(k_i)$')
    plt.legend(loc='best')
    plt.show()

  if args.plot:
    tb.calcDOS(gamma=0.03, npoints=10000, windowsize=1.5)

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax2 = ax1.twinx()
    for ispin in range(tb.spins):
      ax1.plot(tb.dosaxis, tb.dos[ispin], label='dos', color='blue', lw=2)
      ax2.plot(tb.dosaxis, tb.nos[ispin], label='nos', color='red', lw=2)

    ax1.axvline(x=tb.mu, color='black', lw=1, ls='-')
    ax1.set_ylim(ymin=0)
    ax1.set_ylabel(r'$\mathrm{dos}$')
    ax1.set_xlabel(r'$\mu$ [eV]')
    ax1.legend(loc='center left')

    ax2.axhline(y=tb.charge, color='black', lw=1, ls='-')
    ax2.set_ylim(ymin=0)
    ax2.set_ylabel(r'$\mathrm{nos}$')
    ax2.legend(loc='center right')

    plt.show()

if __name__ == '__main__':
  main()

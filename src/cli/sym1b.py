import os
import argparse

from dga import symmetrize as sym


def create_dga_argparser(input_file='Vertex.hdf5', output_file='g4iw_sym.hdf5', path=os.getcwd() + '/'):
    ''' Set up an argument parser for the symmetrize script. '''
    parser = argparse.ArgumentParser()
    parser.add_argument('-if', '--input_file', nargs='?', default=input_file, type=str, help=' Config file name. ')
    parser.add_argument('-of', '--output_file', nargs='?', default=output_file, type=str, help=' Config file name. ')
    parser.add_argument('-p', '--path', nargs='?', default=path, type=str, help=' Path to the config file. ')
    return parser


def main():
    parser = create_dga_argparser()
    args = parser.parse_args()
    worm = 'worm-last'
    print('args',args)
    conf = {'nineq': 1, 'target': '3freq', 'sym_type': 'o',
            'outfile':  args.path + args.output_file, 'Nbands': [1, 1], 'infile': args.path + args.input_file,
            'sym': [[[1]]], 'worm_group': worm}

    sym.main(conf)

if __name__ == '__main__':
    main()
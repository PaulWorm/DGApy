import os
import subprocess

def run_mpi_test(test_fname,n_cores=2):
    '''
        Run MPI test using mpirun.
    '''
    command = ['mpirun', '-np', f'{n_cores}', 'python', test_fname]
    subprocess.run(command)

def run_test(program, test_fname):
    '''
        Run test using python.
    '''
    if type(program) == list:
        command = [*program, test_fname]
    else:
        command = [program, test_fname]
    print('--------------------------------')
    print('Starting Test: ' + ''.join(command))
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, universal_newlines=True)
    # subprocess.run(command)

    # Print subprocess output as it happens
    print(result.stdout)
    print(result.stderr)
    if result.stderr != '':
        print('Test Failed: ' + ''.join(command))
        print('--------------------------------')
        exit(1)

def test_linting():
    '''
        Run pylint on the entire project.
    '''
    # Change directory to the root of the project:
    script_path = os.path.dirname(os.path.abspath(__file__))
    os.chdir(os.path.join(script_path, '../../'))

    # Run pylint on all tracked files:
    program = 'pylint'
    base_path = './src/dga/'
    run_test(program, base_path + 'matsubara_frequencies.py')
    run_test(program, base_path + 'brillouin_zone.py')
    run_test(program, base_path + 'wannier.py')
    run_test(program, base_path + 'two_point.py')
    run_test(program, base_path + 'bubble.py')
    run_test(program, base_path + 'local_four_point.py')
    run_test(program, base_path + 'ornstein_zernicke_function.py')
    run_test(program, base_path + 'w2dyn_aux_dga.py')
    run_test(program, base_path + 'dga_io.py')
    run_test(program, base_path + 'loggers.py')
    run_test(program, base_path + 'mpi_aux.py')
    run_test(program, base_path + 'four_point.py')
    run_test(program, base_path + 'analytic_continuation.py')
    run_test(program, base_path + 'pairing_vertex.py')
    run_test(program, base_path + 'eliashberg_equation.py')
    run_test(program, base_path + 'lambda_correction.py')
    run_test(program, base_path + 'optics.py')

    # go back to the original directory:
    os.chdir(os.path.join(script_path))

def run_unit_tests():
    '''
        Run unit tests for the entire project.
    '''
    # Change directory to the root of the project:
    script_path = os.path.dirname(os.path.abspath(__file__))
    os.chdir(os.path.join(script_path, '../../'))

    # Run unit tests on all tracked files:
    program = 'python'
    base_path = './tests/'
    run_test(program, base_path + 'test_matsubara_frequencies.py')
    run_test(program, base_path + 'test_brillouin_zone.py')
    run_test(program, base_path + 'test_wannier.py')
    run_test(program, base_path + 'test_two_point.py')
    run_test(program, base_path + 'test_eliashberg.py')
    run_test(program, base_path + 'test_bubble.py')
    run_test(program, base_path + 'test_local_four_point.py')
    run_mpi_test('./tests/test_mpi_aux.py', n_cores=2)
    run_test(program, base_path + 'test_four_point.py')
    run_test(program, base_path + 'test_analytic_continuation.py')
    run_test(program, base_path + 'test_pairing_vertex.py')
    run_test(program, base_path + 'test_optics.py')

    # go back to the original directory:
    os.chdir(script_path)

def main():
    test_linting()
    run_unit_tests()

if __name__ == '__main__':
    main()

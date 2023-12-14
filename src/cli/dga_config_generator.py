import os
import shutil

DEFAULT_CONFIG_FILE = 'dga_config.yaml'

def main():
    ''' Copy the default config file to the current directory.
    '''
    # save the execution directory:
    curr_dir = os.getcwd()

    # Change directory to the config_templates directory:
    script_path = os.path.dirname(os.path.abspath(__file__))
    os.chdir(os.path.join(script_path, '../../config_templates/'))
    config_file_dir = os.getcwd()

    # copy the default config file to the current directory:
    src_path = config_file_dir + '/' + DEFAULT_CONFIG_FILE
    dst_path = curr_dir + '/' + DEFAULT_CONFIG_FILE
    shutil.copy(src_path, dst_path)

    # change back to the original folder:
    os.chdir(curr_dir)



if __name__ == '__main__':
    main()

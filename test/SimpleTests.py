import ruamel.yaml

def to_yaml():
    my_dict = {
    'table_1' : ['col_1', 'col_2', 'col_3', 'col_4']
    }

    my_file_path = "./"

    with open(f"{my_file_path}/structure.yml", "w") as file:
        # yaml.dump need a dict and a file handler as parameter
        yaml = ruamel.yaml.YAML()
        yaml.indent(sequence=4, offset=2)
        yaml.dump(my_dict, file)

if __name__ == '__main__':
    to_yaml()
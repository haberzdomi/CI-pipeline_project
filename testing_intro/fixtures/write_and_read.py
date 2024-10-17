def create_empty_file(filename):
    with open(filename, "w") as file:
        pass

def read_file(filename):
    with open(filename, "r") as file:
        return file.read()
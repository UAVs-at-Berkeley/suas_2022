import os
from rembg import remove

PATH_OF_SCRIPT = os.path.dirname(os.path.abspath(__file__))
INPUT_DIRECTORY = PATH_OF_SCRIPT + '\\input'
OUTPUT_DIRECTORY = PATH_OF_SCRIPT + '\\output'

if __name__ == '__main__':
    files = os.listdir(INPUT_DIRECTORY)
    for file in files:
        with open(INPUT_DIRECTORY + '//' + file, 'rb') as i:
                new_file_name = OUTPUT_DIRECTORY + '\\no_background_' + file
                with open(new_file_name, 'wb') as o:
                    input = i.read()
                    output = remove(input)
                    o.write(output)
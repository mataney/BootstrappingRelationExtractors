import os

def read_file(file_path, remove_duplicates=False):
    with open(file_path, 'r') as file:
        ret = file.readlines()
    ret = [s.strip() for s in ret]
    if remove_duplicates:
        ret = list(set(ret))
    if '' in ret: ret.remove('')

    return ret

def write_to_file(filtered_sentences, model_folder, generation_file, prefix=''):
    output_file_path = os.path.join(model_folder, prefix + generation_file)
    with open(output_file_path, 'w') as file:
        for sent in filtered_sentences:
            file.write(f"{sent}\n")

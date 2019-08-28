from constants import num_samples


def read_dir(inp_lang, tar_lang):
    with open(inp_lang, 'r', encoding='utf-8') as f:
        lines = f.read().split('\n')
    return True


def read_file(path):
    inp = []
    tar = []
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            inp.append(line.split('\t')[0].lower())
            tar.append(line.split('\t')[1].lower())
    return inp, tar


def read_corpus(path, fformat, nlevel, num_samples=num_samples, **kwargs):
    start_sequence = kwargs.get('start_sequence', '\t')
    end_sequence = kwargs.get('end_sequence', '\n')
    input_characters = set()
    target_characters = set()

    if fformat == "dir":
        input_path = kwargs.get('inp_lang_path')
        target_path = kwargs.get('tar_lang_path')
        inp, tar = read_dir(input_path, target_path)
    else:
        inp, tar = read_file(path)
    input_texts = []
    target_texts = []

    for k, v in enumerate(inp[: min(num_samples, len(input_texts) - 1)]):
        input_text, target_text = v, tar[k]
        target_text = start_sequence + target_text + end_sequence
        target_text = '\t' + target_text + '\n'
        input_texts.append(input_text)
        target_texts.append(target_text)
        for char in input_text:
            if char not in input_characters:
                input_characters.add(char)
        for char in target_text:
            if char not in target_characters:
                target_characters.add(char)

    return sorted(list(input_characters)), sorted(list(target_characters)), input_texts, target_texts

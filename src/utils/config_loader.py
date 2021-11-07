config_path = '../config.config'

with open(config_path) as f:
    lines = list(f)
    dict_key_word = []
    for line in lines:
        key, word = line.strip().split('=')
        print(str(key) + ': ' + str(word))
        dict_key_word.append((key, word))
    dictionary = dict(dict_key_word)
    threads_loading = int(dictionary['threads_loading'])
    cache = int(dictionary['cache'])
    batch_size = int(dictionary['batch_size'])
    model_path = dictionary['model_path']
    model_path_logs = dictionary['model_path_logs']
    train_data = dictionary['train_data']
    test_data = dictionary['test_data']
    valid_data = dictionary['valid_data']
    one_test_data_dir = dictionary['one_test_data_dir']
    output_data_dir = dictionary['output_data_dir']
    output_model_dir = dictionary['output_model_dir']

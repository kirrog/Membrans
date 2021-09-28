# from utils.augmentations itransformsmport augment_dataset
# from clearer.datasets_loader import load_clearer_dataset_to_numpy_table, save_clearer_dataset_to_numpy_table, load_clearer_dataset_predicts, load_clearer_dataset_answers

from src.utils.augmentations import augment_dataset
from src.clearer.datasets_loader import load_clearer_dataset_to_numpy_table, save_clearer_dataset_to_numpy_table, \
    load_clearer_dataset_predicts, load_clearer_dataset_answers

# augments_path = '/content/drive/MyDrive/Membrans/dataset/numpys/augmentations'

augments_path = '../dataset/numpys/augmentations'
predictors_path = '/pred_'
answers_path = '/answ_'

augment_size = 10


def create_augment(number, images, masks):
    predictors, answers = augment_dataset(images, masks)
    save_clearer_dataset_to_numpy_table(predictors, (augments_path + predictors_path + str(number) + '.npy'))
    save_clearer_dataset_to_numpy_table(answers, (augments_path + answers_path + str(number) + '.npy'))


def create_augmented_datasets():
    answers_orig = load_clearer_dataset_answers()
    predictors_orig = load_clearer_dataset_predicts()
    for i in range(augment_size):
        print('\nCreating ' + str(i) + ' augmented dataset')
        create_augment(i, predictors_orig, answers_orig)


def get_augment_dataset(num):
    return load_clearer_dataset_to_numpy_table(augments_path + predictors_path + str(num) + '.npy')

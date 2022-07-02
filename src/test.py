import sys

import tqdm as tqdm

from src.generator.dataset_generator_providers.dataset_generator_providers import \
    generator_dataset_pair_generator_parallel_getter
from src.utils import config_loader

g = generator_dataset_pair_generator_parallel_getter(config_loader.train_data)()
iteration = 0
for i in g:
    iteration += 1
    sys.stdout.write(f"\r{iteration:05d}")

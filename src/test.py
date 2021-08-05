from src.clearer.clearer_model import load_model
from src.utils.load_test_dataset import load_clearer_teset_set
import matplotlib.pyplot as plt

model = load_model()
data = load_clearer_teset_set()
results = model.predict(data)

im1 = results[0,...,0]
plt.imshow((im1)>0.6)
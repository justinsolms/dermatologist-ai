# %%
import numpy as np
from dermatologist.data import Data
from dermatologist.models import Model

# %%
data_obj = Data(unittest_size=0.015)

# %%
model = Model(data_obj, epochs=10, batch_size=10)

# %%
model.fit()

# %%
model.predict()

# %%
model.report()


# %%

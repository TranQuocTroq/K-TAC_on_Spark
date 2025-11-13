# src/model_trainer/__init__.py

# File này cũng có thể để trống.
# Nó báo cho Python biết 'model_trainer' là một package con.

# Tùy chọn nâng cao: Bạn có thể "shortcut" các class quan trọng
# để import dễ hơn sau này.
# Ví dụ, thay vì gõ:
# from src.model_trainer.architecture import FocusOnSpark
# Bạn có thể gõ:
# from src.model_trainer import FocusOnSpark

from .architecture import FocusOnSpark
from .dataset import WSIFocusDataset
from .engine import train_one_epoch, evaluate


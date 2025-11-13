# src/__init__.py

# File này có thể để trống.
# Nó báo cho Python biết 'src' là một package.
# Điều này giúp bạn import các module con như:
# from src.model_trainer import architecture
# from src.spark_pipeline import feature_extractor

import os
import sys

# Thêm thư mục gốc của project (cha của 'src') vào PYTHONPATH
# Giúp các script con dễ dàng import lẫn nhau.
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
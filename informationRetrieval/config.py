# ===== config.py (增强版) =====
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)

PROCESSED_DATA_FILE = DATA_DIR / "processed_amazon_data.csv"

API_HOST = "localhost"
API_PORT = 8000

SEARCH_CONFIG = {
    'max_features': 5000,
    'max_results': 100,
    'min_score_threshold': 0.01,
    'default_top_k': 10
}

# 拼写纠错配置
SPELL_CORRECTION_CONFIG = {
    'max_edit_distance': 2,          # 最大编辑距离
    'min_word_length': 3,            # 最小词长度
    'min_word_frequency': 1,         # 最小词频
    'max_candidates': 5              # 最大候选词数
}

# 同义词扩展配置
SYNONYM_CONFIG = {
    'max_expansions_per_word': 2,    # 每个词最大扩展数
    'enable_reverse_lookup': True,   # 启用反向查找
    'synonym_weight': 0.8            # 同义词权重
}

# 搜索策略配置
SEARCH_STRATEGY_CONFIG = {
    'spell_correction_threshold': 0.1,    # 拼写纠错阈值
    'synonym_expansion_threshold': 0.1,   # 同义词扩展阈值
    'result_count_threshold': 0.5,        # 结果数量阈值（相对于top_k）
    'score_improvement_threshold': 1.2    # 分数改进阈值
}
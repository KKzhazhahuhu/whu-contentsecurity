import pandas as pd
import numpy as np
import json
import re
import ast
from typing import List, Dict, Any, Optional
from datasets import load_dataset
import logging
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
import config


class DataPreprocessor:
    def __init__(self):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def load_data(self) -> pd.DataFrame:
        """加载Amazon数据集"""
        self.logger.info("加载Amazon数据集...")
        ds = load_dataset("randomath/Amazon-combined")
        df = pd.DataFrame(ds['train'])
        self.logger.info(f"加载完成: {len(df)} 条记录")
        return df

    def clean_price(self, price_str) -> Optional[float]:
        """清洗价格数据"""
        if pd.isna(price_str) or price_str is None:
            return None

        try:
            if isinstance(price_str, (int, float)):
                return float(price_str) if price_str > 0 else None

            price_str = str(price_str).strip()
            price_clean = re.sub(r'[^\d.,]', '', price_str)

            if not price_clean:
                return None

            # 处理价格格式
            if ',' in price_clean and '.' in price_clean:
                parts = price_clean.split('.')
                if len(parts) == 2 and len(parts[1]) <= 2:
                    price_clean = price_clean.replace(',', '')
            elif ',' in price_clean:
                if price_clean.count(',') == 1 and len(price_clean.split(',')[1]) <= 2:
                    price_clean = price_clean.replace(',', '.')
                else:
                    price_clean = price_clean.replace(',', '')

            price = float(price_clean)
            return price if 0 < price < 1000000 else None
        except:
            return None

    def clean_description(self, desc) -> str:
        """清洗描述数据"""
        if pd.isna(desc) or desc is None:
            return ""

        try:
            if isinstance(desc, str):
                try:
                    desc = ast.literal_eval(desc)
                except:
                    pass

            if isinstance(desc, list):
                full_desc = " ".join(str(item) for item in desc if item)
            else:
                full_desc = str(desc)

            # 移除HTML标签和多余空格
            clean_desc = re.sub(r'<[^>]+>', '', full_desc)
            clean_desc = re.sub(r'\s+', ' ', clean_desc).strip()
            return clean_desc[:500]  # 限制长度
        except:
            return ""

    def parse_list_field(self, field_value) -> List[str]:
        """解析列表字段"""
        if pd.isna(field_value) or field_value is None:
            return []

        try:
            if isinstance(field_value, list):
                return [str(item).strip() for item in field_value if item]

            if isinstance(field_value, str):
                try:
                    parsed = ast.literal_eval(field_value)
                    if isinstance(parsed, list):
                        return [str(item).strip() for item in parsed if item]
                except:
                    return [item.strip() for item in field_value.split(',') if item.strip()]
        except:
            pass

        return []

    def create_search_text(self, row: pd.Series) -> str:
        """创建搜索文本"""
        text_parts = []

        # 标题
        title = row.get('title', '')
        if title and str(title) != 'nan':
            text_parts.append(str(title))

        # 主分类
        main_category = row.get('main_category', '')
        if main_category and str(main_category) != 'nan':
            text_parts.append(str(main_category))

        # 描述
        description = row.get('description_clean', '')
        if description:
            text_parts.append(description)

        # 分类列表
        categories = row.get('categories_clean', [])
        if isinstance(categories, list):
            text_parts.extend(categories)

        return " ".join(text_parts).lower()

    def process_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """处理数据"""
        self.logger.info("开始数据预处理...")
        processed_df = df.copy()

        # 清洗价格
        processed_df['price_clean'] = processed_df['price'].apply(self.clean_price)

        # 清洗描述
        processed_df['description_clean'] = processed_df['description'].apply(self.clean_description)

        # 解析分类
        processed_df['categories_clean'] = processed_df['categories'].apply(self.parse_list_field)

        # 创建搜索文本
        processed_df['search_text'] = processed_df.apply(self.create_search_text, axis=1)

        # 添加产品ID
        processed_df['product_id'] = range(len(processed_df))

        # 过滤无效数据
        processed_df = processed_df[
            processed_df['title'].notna() &
            (processed_df['title'].astype(str) != 'nan') &
            (processed_df['title'].astype(str).str.strip() != '')
            ].copy()

        processed_df = processed_df.reset_index(drop=True)
        processed_df['product_id'] = range(len(processed_df))

        self.logger.info(f"数据预处理完成: {len(processed_df)} 条记录")
        return processed_df

    def save_data(self, df: pd.DataFrame, filename: str = None):
        """保存数据"""
        if filename is None:
            filename = config.PROCESSED_DATA_FILE

        columns_to_save = [
            'product_id', 'title', 'main_category', 'categories_clean',
            'description_clean', 'price_clean', 'average_rating',
            'rating_number', 'search_text', 'asin'
        ]

        available_columns = [col for col in columns_to_save if col in df.columns]
        save_df = df[available_columns].copy()
        save_df.to_csv(filename, index=False, encoding='utf-8')
        self.logger.info(f"数据已保存到 {filename}")
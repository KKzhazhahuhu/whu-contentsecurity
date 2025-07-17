# ===== src/search_engine.py (最终工作版本) =====
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Set
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import logging
from collections import defaultdict, Counter
import sys
import os
from pathlib import Path

# 修复：更详细的导入检测
SEMANTIC_SEARCH_AVAILABLE = False
SentenceTransformer = None
Transformer = None
Pooling = None

try:
    from sentence_transformers import SentenceTransformer
    from sentence_transformers.models import Transformer, Pooling

    SEMANTIC_SEARCH_AVAILABLE = True
    print("✅ sentence-transformers 导入成功")
except ImportError as e:
    print(f"⚠️  sentence-transformers 导入失败: {e}")
    SEMANTIC_SEARCH_AVAILABLE = False

sys.path.append(str(Path(__file__).parent.parent))
import config


class SemanticSearchEngine:
    """语义搜索引擎"""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = None
        self.semantic_embeddings = None
        self.logger = logging.getLogger(__name__)

        if SEMANTIC_SEARCH_AVAILABLE and SentenceTransformer is not None:
            try:
                # 检查本地模型
                local_model_path = Path(__file__).parent.parent / "models" / model_name

                if local_model_path.exists() and local_model_path.is_dir():
                    # 检查必要的模型文件是否存在
                    required_files = ['config.json', 'pytorch_model.bin', 'tokenizer.json']
                    missing_files = [f for f in required_files if not (local_model_path / f).exists()]

                    if not missing_files:
                        print(f"发现完整的本地模型: {local_model_path}")

                        # 使用手动构建方法（已验证成功）
                        success = False

                        try:
                            print("使用手动构建方法加载模型...")

                            # 手动创建 Transformer 模型
                            transformer = Transformer(str(local_model_path))
                            print("✅ Transformer 组件加载成功")

                            # 创建 Pooling 层（使用 mean pooling）
                            pooling = Pooling(
                                transformer.get_word_embedding_dimension(),
                                pooling_mode='mean'
                            )
                            print("✅ Pooling 组件创建成功")

                            # 组合成 SentenceTransformer
                            self.model = SentenceTransformer(modules=[transformer, pooling])
                            print(f"✅ 手动构建语义搜索模型成功: {model_name}")
                            success = True

                        except Exception as e1:
                            print(f"❌ 手动构建失败: {e1}")

                        if not success:
                            print("❌ 本地模型加载失败")
                            raise Exception("本地模型加载失败")

                    else:
                        print(f"❌ 本地模型不完整，缺少文件: {missing_files}")
                        raise Exception("本地模型文件不完整")
                else:
                    print(f"本地模型目录不存在: {local_model_path}")
                    raise Exception("本地模型目录不存在")

                self.logger.info(f"语义搜索模型加载成功: {model_name}")

            except Exception as e:
                self.logger.warning(f"本地模型加载失败: {e}")
                print(f"❌ 本地模型加载失败: {e}")
                print("💡 尝试在线加载...")

                # 尝试在线加载
                try:
                    print(f"尝试在线加载模型: {model_name}")
                    self.model = SentenceTransformer(model_name)
                    print(f"✅ 在线语义搜索模型加载成功: {model_name}")
                except Exception as e2:
                    print(f"❌ 在线加载也失败: {e2}")
                    print("💡 系统将使用关键词搜索模式")
                    self.model = None
        else:
            self.logger.warning("sentence-transformers不可用，语义搜索功能不可用")
            print("⚠️  sentence-transformers不可用，语义搜索功能不可用")

    def build_embeddings(self, documents: List[str]):
        """构建文档的语义嵌入"""
        if not self.model:
            print("❌ 语义模型未加载，跳过嵌入构建")
            return None

        try:
            self.logger.info("开始构建语义嵌入...")
            print("🔄 开始构建语义嵌入...")

            # 过滤空文档
            valid_docs = [doc if doc and doc.strip() else "empty document" for doc in documents]
            print(f"   处理文档数量: {len(valid_docs)}")

            # 批量编码，使用较小的batch_size避免内存问题
            self.semantic_embeddings = self.model.encode(
                valid_docs,
                batch_size=16,  # 减小batch_size
                show_progress_bar=True,
                convert_to_numpy=True,
                normalize_embeddings=True  # 标准化嵌入
            )

            self.logger.info(f"语义嵌入构建完成: {self.semantic_embeddings.shape}")
            print(f"✅ 语义嵌入构建完成: {self.semantic_embeddings.shape}")
            return self.semantic_embeddings
        except Exception as e:
            self.logger.error(f"构建语义嵌入失败: {e}")
            print(f"❌ 构建语义嵌入失败: {e}")
            return None

    def search(self, query: str, top_k: int = 10) -> List[Tuple[int, float]]:
        """语义搜索"""
        if not self.model or self.semantic_embeddings is None:
            return []

        try:
            # 编码查询
            query_embedding = self.model.encode([query], convert_to_numpy=True, normalize_embeddings=True)

            # 计算语义相似度
            similarities = cosine_similarity(query_embedding, self.semantic_embeddings).flatten()

            # 获取top-k结果
            top_indices = np.argsort(similarities)[::-1][:top_k]

            return [(int(idx), float(similarities[idx])) for idx in top_indices
                    if similarities[idx] > 0.1]  # 过滤低相似度结果
        except Exception as e:
            self.logger.error(f"语义搜索失败: {e}")
            return []


class SpellCorrector:
    """拼写纠错器"""

    def __init__(self, max_distance: int = 2):
        self.max_distance = max_distance
        self.word_freq = Counter()
        self.vocabulary = set()

    def build_vocabulary(self, documents: List[str]):
        """从文档中构建词汇表"""
        for doc in documents:
            words = re.findall(r'\b\w+\b', doc.lower())
            for word in words:
                if len(word) > 2:
                    self.word_freq[word] += 1
                    self.vocabulary.add(word)

        min_freq = max(1, len(documents) // 1000)
        self.vocabulary = {word for word, freq in self.word_freq.items() if freq >= min_freq}

        logging.info(f"构建词汇表完成，包含 {len(self.vocabulary)} 个词汇")

    def levenshtein_distance(self, s1: str, s2: str) -> int:
        """计算编辑距离"""
        if len(s1) < len(s2):
            return self.levenshtein_distance(s2, s1)

        if len(s2) == 0:
            return len(s1)

        previous_row = list(range(len(s2) + 1))
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row

        return previous_row[-1]

    def get_candidates(self, word: str) -> List[Tuple[str, int]]:
        """获取拼写候选词"""
        if word in self.vocabulary:
            return [(word, 0)]

        candidates = []
        word_lower = word.lower()

        for vocab_word in self.vocabulary:
            if abs(len(vocab_word) - len(word_lower)) > self.max_distance:
                continue

            distance = self.levenshtein_distance(word_lower, vocab_word)
            if distance <= self.max_distance:
                candidates.append((vocab_word, distance))

        candidates.sort(key=lambda x: (x[1], -self.word_freq.get(x[0], 0)))
        return candidates[:5]

    def correct_query(self, query: str) -> Tuple[str, List[str]]:
        """纠正查询语句"""
        words = re.findall(r'\b\w+\b', query.lower())
        corrected_words = []
        corrections = []

        for word in words:
            candidates = self.get_candidates(word)
            if candidates:
                best_word, distance = candidates[0]
                corrected_words.append(best_word)
                if distance > 0:
                    corrections.append(f"{word} → {best_word}")
            else:
                corrected_words.append(word)

        corrected_query = ' '.join(corrected_words)
        return corrected_query, corrections


class SynonymExpander:
    """同义词扩展器"""

    def __init__(self):
        self.synonyms = {
            # 电子产品
            'phone': ['smartphone', 'mobile', 'cellphone', 'iphone', 'android'],
            'smartphone': ['phone', 'mobile', 'cellphone'],
            'laptop': ['notebook', 'computer', 'pc'],
            'computer': ['laptop', 'desktop', 'pc'],
            'tablet': ['ipad', 'pad'],

            # 服装
            'shirt': ['tshirt', 'blouse', 'top'],
            'pants': ['trousers', 'jeans'],
            'shoes': ['sneakers', 'boots', 'footwear'],

            # 家居
            'chair': ['seat', 'stool'],
            'table': ['desk'],
            'lamp': ['light', 'lighting'],

            # 书籍
            'book': ['novel', 'textbook', 'ebook'],
            'magazine': ['journal', 'periodical'],

            # 通用词汇
            'good': ['great', 'excellent', 'quality'],
            'cheap': ['affordable', 'budget', 'inexpensive'],
            'expensive': ['premium', 'luxury', 'high-end'],
            'small': ['mini', 'compact', 'portable'],
            'large': ['big', 'huge', 'xl'],
        }

        self.reverse_synonyms = {}
        for word, syns in self.synonyms.items():
            for syn in syns:
                if syn not in self.reverse_synonyms:
                    self.reverse_synonyms[syn] = []
                self.reverse_synonyms[syn].append(word)

    def expand_query(self, query: str, max_expansions: int = 2) -> str:
        """扩展查询语句"""
        words = re.findall(r'\b\w+\b', query.lower())
        expanded_terms = set(words)

        for word in words:
            if word in self.synonyms:
                expanded_terms.update(self.synonyms[word][:max_expansions])
            if word in self.reverse_synonyms:
                expanded_terms.update(self.reverse_synonyms[word][:max_expansions])

        return ' '.join(expanded_terms)


class HybridSearchEngine:
    """混合检索搜索引擎"""

    def __init__(self, data: pd.DataFrame, semantic_model: str = "all-MiniLM-L6-v2"):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        self.data = data.copy()

        # 关键词搜索组件
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None

        # 语义搜索组件
        self.semantic_engine = SemanticSearchEngine(semantic_model)

        # 拼写纠错和同义词扩展
        self.spell_corrector = SpellCorrector()
        self.synonym_expander = SynonymExpander()

        # 混合检索权重
        self.keyword_weight = 0.6
        self.semantic_weight = 0.4

        self._build_index()

    def _build_index(self):
        """构建混合搜索索引"""
        self.logger.info("构建混合搜索索引...")

        # 准备文档
        documents = []
        for _, row in self.data.iterrows():
            search_text = row.get('search_text', '')
            if pd.notna(search_text) and search_text.strip():
                documents.append(str(search_text))
            else:
                documents.append("")

        if not documents or all(not doc.strip() for doc in documents):
            raise ValueError("没有有效的搜索文档")

        # 构建拼写纠错词汇表
        self.spell_corrector.build_vocabulary(documents)

        # 构建TF-IDF索引
        self.logger.info("构建TF-IDF索引...")
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=config.SEARCH_CONFIG['max_features'],
            stop_words='english',
            lowercase=True,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.8
        )
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(documents)

        # 构建语义嵌入索引
        if self.semantic_engine.model:
            self.semantic_engine.build_embeddings(documents)
        else:
            print("⚠️  语义搜索不可用，将使用纯关键词搜索模式")

        self.logger.info("混合搜索索引构建完成")

    def search(self, query: str, top_k: int = None, filters: Dict = None,
               enable_spell_correction: bool = True,
               enable_synonym_expansion: bool = True,
               use_hybrid: bool = True,
               keyword_weight: float = 0.6) -> Dict[str, Any]:
        """混合检索搜索"""
        if not query or not query.strip():
            return {'results': [], 'query_info': {}}

        if top_k is None:
            top_k = config.SEARCH_CONFIG['default_top_k']

        # 更新权重
        self.keyword_weight = keyword_weight
        self.semantic_weight = 1.0 - keyword_weight

        try:
            original_query = query.strip()
            processed_query = original_query.lower()
            query_info = {
                'original_query': original_query,
                'processed_query': processed_query,
                'corrections': [],
                'expansions': [],
                'search_strategy': 'hybrid' if (use_hybrid and self.semantic_engine.model) else 'keyword_only',
                'keyword_weight': self.keyword_weight,
                'semantic_weight': self.semantic_weight
            }

            # 第一步：尝试混合搜索
            if use_hybrid and self.semantic_engine.model:
                results = self._execute_hybrid_search(processed_query, top_k, filters)
            else:
                results = self._execute_keyword_search(processed_query, top_k, filters)
                query_info['search_strategy'] = 'keyword_only'

            # 如果结果不够好，尝试拼写纠错
            if (enable_spell_correction and
                    (not results or len(results) < top_k // 2 or
                     (results and results[0]['score'] < 0.1))):

                corrected_query, corrections = self.spell_corrector.correct_query(processed_query)
                if corrections:
                    if use_hybrid and self.semantic_engine.model:
                        corrected_results = self._execute_hybrid_search(corrected_query, top_k, filters)
                    else:
                        corrected_results = self._execute_keyword_search(corrected_query, top_k, filters)

                    if corrected_results and (not results or corrected_results[0]['score'] > results[0]['score']):
                        results = corrected_results
                        query_info['processed_query'] = corrected_query
                        query_info['corrections'] = corrections
                        query_info['search_strategy'] += '_spell_corrected'

            # 如果结果仍然不够好，尝试同义词扩展
            if (enable_synonym_expansion and
                    (not results or len(results) < top_k // 2 or
                     (results and results[0]['score'] < 0.1))):

                expanded_query = self.synonym_expander.expand_query(query_info['processed_query'])
                if expanded_query != query_info['processed_query']:
                    if use_hybrid and self.semantic_engine.model:
                        expanded_results = self._execute_hybrid_search(expanded_query, top_k, filters)
                    else:
                        expanded_results = self._execute_keyword_search(expanded_query, top_k, filters)

                    if expanded_results and (not results or expanded_results[0]['score'] > results[0]['score'] * 0.8):
                        combined_results = self._merge_results(results, expanded_results, top_k)
                        if len(combined_results) > len(results):
                            results = combined_results
                            query_info['expansions'] = expanded_query.split()
                            query_info['search_strategy'] += '_synonym_expanded'

            return {
                'results': results,
                'query_info': query_info,
                'total_count': len(results)
            }

        except Exception as e:
            self.logger.error(f"搜索失败: {e}")
            return {'results': [], 'query_info': {'error': str(e)}}

    def _execute_hybrid_search(self, query: str, top_k: int, filters: Dict = None) -> List[Dict[str, Any]]:
        """执行混合检索搜索"""
        if not query.strip():
            return []

        # 关键词搜索结果
        keyword_results = self._get_keyword_similarities(query)

        # 语义搜索结果
        semantic_results = self._get_semantic_similarities(query)

        # 混合得分计算
        hybrid_scores = {}
        max_results = min(len(self.data), top_k * 3)

        # 处理关键词搜索结果
        for idx, score in keyword_results[:max_results]:
            hybrid_scores[idx] = score * self.keyword_weight

        # 处理语义搜索结果
        for idx, score in semantic_results[:max_results]:
            if idx in hybrid_scores:
                hybrid_scores[idx] += score * self.semantic_weight
            else:
                hybrid_scores[idx] = score * self.semantic_weight

        # 排序并构建结果
        sorted_results = sorted(hybrid_scores.items(), key=lambda x: x[1], reverse=True)

        results = []
        for idx, score in sorted_results[:top_k * 2]:
            if score > config.SEARCH_CONFIG['min_score_threshold']:
                product = self.data.iloc[idx]

                result = {
                    'product_id': int(product.get('product_id', idx)),
                    'title': str(product.get('title', '')),
                    'main_category': str(product.get('main_category', '')),
                    'description': str(product.get('description_clean', '')),
                    'price': self._safe_float(product.get('price_clean')),
                    'average_rating': self._safe_float(product.get('average_rating')),
                    'rating_number': self._safe_int(product.get('rating_number')),
                    'asin': str(product.get('asin', '')),
                    'score': float(score),
                    'search_type': 'hybrid'
                }
                results.append(result)

        # 应用过滤器
        if filters:
            results = self._apply_filters(results, filters)

        return results[:top_k]

    def _execute_keyword_search(self, query: str, top_k: int, filters: Dict = None) -> List[Dict[str, Any]]:
        """执行关键词搜索"""
        keyword_results = self._get_keyword_similarities(query)

        results = []
        for idx, score in keyword_results[:top_k * 2]:
            if score > config.SEARCH_CONFIG['min_score_threshold']:
                product = self.data.iloc[idx]

                result = {
                    'product_id': int(product.get('product_id', idx)),
                    'title': str(product.get('title', '')),
                    'main_category': str(product.get('main_category', '')),
                    'description': str(product.get('description_clean', '')),
                    'price': self._safe_float(product.get('price_clean')),
                    'average_rating': self._safe_float(product.get('average_rating')),
                    'rating_number': self._safe_int(product.get('rating_number')),
                    'asin': str(product.get('asin', '')),
                    'score': float(score),
                    'search_type': 'keyword'
                }
                results.append(result)

        if filters:
            results = self._apply_filters(results, filters)

        return results[:top_k]

    def _get_keyword_similarities(self, query: str) -> List[Tuple[int, float]]:
        """获取关键词相似度"""
        query_vector = self.tfidf_vectorizer.transform([query])
        similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
        top_indices = np.argsort(similarities)[::-1]

        return [(int(idx), float(similarities[idx])) for idx in top_indices
                if similarities[idx] > 0]

    def _get_semantic_similarities(self, query: str) -> List[Tuple[int, float]]:
        """获取语义相似度"""
        if not self.semantic_engine.model:
            return []

        return self.semantic_engine.search(query, top_k=len(self.data))

    def _merge_results(self, results1: List[Dict], results2: List[Dict], top_k: int) -> List[Dict]:
        """合并搜索结果"""
        seen_ids = set()
        merged = []

        all_results = results1 + results2
        all_results.sort(key=lambda x: x['score'], reverse=True)

        for result in all_results:
            product_id = result['product_id']
            if product_id not in seen_ids:
                seen_ids.add(product_id)
                merged.append(result)
                if len(merged) >= top_k:
                    break

        return merged

    def _safe_float(self, value) -> Optional[float]:
        """安全转换为浮点数"""
        if pd.isna(value) or value is None:
            return None
        try:
            return float(value)
        except:
            return None

    def _safe_int(self, value) -> Optional[int]:
        """安全转换为整数"""
        if pd.isna(value) or value is None:
            return None
        try:
            return int(value)
        except:
            return None

    def _apply_filters(self, results: List[Dict], filters: Dict) -> List[Dict]:
        """应用过滤器"""
        filtered_results = results.copy()

        if 'category' in filters:
            target_category = str(filters['category']).lower()
            filtered_results = [
                r for r in filtered_results
                if target_category in r['main_category'].lower()
            ]

        if 'price_min' in filters:
            price_min = filters['price_min']
            filtered_results = [
                r for r in filtered_results
                if r['price'] is not None and r['price'] >= price_min
            ]

        if 'price_max' in filters:
            price_max = filters['price_max']
            filtered_results = [
                r for r in filtered_results
                if r['price'] is not None and r['price'] <= price_max
            ]

        return filtered_results

    def get_search_capabilities(self) -> Dict[str, bool]:
        """获取搜索能力信息"""
        return {
            'keyword_search': True,
            'semantic_search': self.semantic_engine.model is not None,
            'spell_correction': True,
            'synonym_expansion': True,
            'hybrid_search': self.semantic_engine.model is not None
        }


# 为了向后兼容，保留原有的SearchEngine类
SearchEngine = HybridSearchEngine
# ===== api/main.py (混合检索版) =====
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import pandas as pd
import uvicorn
import logging
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import config
from src.search_engine import HybridSearchEngine
from src.data_preprocessor import DataPreprocessor

app = FastAPI(title="电商信息检索API (混合检索版)", version="3.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class SearchRequest(BaseModel):
    query: str
    top_k: Optional[int] = 10
    filters: Optional[Dict[str, Any]] = {}
    enable_spell_correction: Optional[bool] = True
    enable_synonym_expansion: Optional[bool] = True
    use_hybrid: Optional[bool] = True  # 启用混合检索
    keyword_weight: Optional[float] = 0.6  # 关键词权重
    semantic_model: Optional[str] = "all-MiniLM-L6-v2"  # 语义模型


class SearchResponse(BaseModel):
    results: List[Dict[str, Any]]
    total_count: int
    query_info: Dict[str, Any]
    message: Optional[str] = None
    search_capabilities: Optional[Dict[str, bool]] = None


# 全局变量
search_engine: Optional[HybridSearchEngine] = None
processed_data: Optional[pd.DataFrame] = None


def initialize_system():
    """初始化系统"""
    global search_engine, processed_data

    try:
        # 检查预处理数据是否存在
        if config.PROCESSED_DATA_FILE.exists():
            processed_data = pd.read_csv(config.PROCESSED_DATA_FILE)
            print(f"加载预处理数据: {len(processed_data)} 条记录")
        else:
            print("开始数据预处理...")
            preprocessor = DataPreprocessor()
            raw_data = preprocessor.load_data()
            processed_data = preprocessor.process_data(raw_data)
            preprocessor.save_data(processed_data)

        # 初始化混合搜索引擎
        print("正在初始化混合搜索引擎...")
        search_engine = HybridSearchEngine(processed_data)

        capabilities = search_engine.get_search_capabilities()
        print("混合搜索引擎初始化完成")
        print(f"搜索能力: {capabilities}")
        return True

    except Exception as e:
        print(f"系统初始化失败: {e}")
        return False


@app.on_event("startup")
async def startup_event():
    """应用启动时初始化"""
    initialize_system()


@app.get("/")
async def root():
    """根路径"""
    capabilities = search_engine.get_search_capabilities() if search_engine else {}

    return {
        "message": "电商信息检索API (混合检索版)",
        "version": "3.0.0",
        "features": [
            "TF-IDF关键词搜索",
            "语义搜索 (Sentence Transformers)" if capabilities.get('semantic_search') else "语义搜索 (未启用)",
            "混合检索 (关键词+语义)",
            "拼写纠错 (Levenshtein距离)",
            "同义词扩展",
            "多策略智能搜索"
        ],
        "status": "running",
        "system_ready": search_engine is not None,
        "search_capabilities": capabilities
    }


@app.post("/api/search", response_model=SearchResponse)
async def search_products(request: SearchRequest):
    """混合检索搜索产品"""
    if not search_engine:
        raise HTTPException(status_code=503, detail="搜索引擎未初始化")

    if not request.query or not request.query.strip():
        raise HTTPException(status_code=400, detail="查询参数不能为空")

    try:
        # 验证权重参数
        if not (0 <= request.keyword_weight <= 1):
            raise HTTPException(status_code=400, detail="关键词权重必须在0-1之间")

        search_result = search_engine.search(
            query=request.query,
            top_k=request.top_k,
            filters=request.filters,
            enable_spell_correction=request.enable_spell_correction,
            enable_synonym_expansion=request.enable_synonym_expansion,
            use_hybrid=request.use_hybrid,
            keyword_weight=request.keyword_weight
        )

        # 生成用户友好的消息
        message = None
        query_info = search_result.get('query_info', {})

        if query_info.get('corrections'):
            corrections = ', '.join(query_info['corrections'])
            message = f"已为您纠正拼写: {corrections}"
        elif query_info.get('expansions'):
            message = "已使用同义词扩展搜索以获得更多结果"
        elif 'hybrid' in query_info.get('search_strategy', ''):
            message = f"使用混合检索 (关键词权重: {query_info.get('keyword_weight', 0.6):.1f}, 语义权重: {query_info.get('semantic_weight', 0.4):.1f})"
        elif query_info.get('search_strategy') == 'keyword_only':
            message = "使用关键词搜索 (语义搜索不可用)"

        return SearchResponse(
            results=search_result.get('results', []),
            total_count=search_result.get('total_count', 0),
            query_info=query_info,
            message=message,
            search_capabilities=search_engine.get_search_capabilities()
        )

    except Exception as e:
        logging.error(f"搜索失败: {e}")
        raise HTTPException(status_code=500, detail=f"搜索失败: {str(e)}")


@app.get("/health")
async def health_check():
    """健康检查"""
    capabilities = search_engine.get_search_capabilities() if search_engine else {}

    return {
        "status": "healthy" if search_engine else "initializing",
        "version": "3.0.0",
        "data_loaded": processed_data is not None,
        "engine_ready": search_engine is not None,
        "total_products": len(processed_data) if processed_data is not None else 0,
        "search_capabilities": capabilities
    }


@app.get("/api/capabilities")
async def get_search_capabilities():
    """获取搜索能力信息"""
    if not search_engine:
        raise HTTPException(status_code=503, detail="搜索引擎未初始化")

    return search_engine.get_search_capabilities()


@app.get("/api/synonyms")
async def get_synonyms():
    """获取同义词词典"""
    if not search_engine:
        raise HTTPException(status_code=503, detail="搜索引擎未初始化")

    return {
        "synonyms": search_engine.synonym_expander.synonyms,
        "total_words": len(search_engine.synonym_expander.synonyms)
    }


@app.post("/api/spell-check")
async def spell_check(query: str):
    """拼写检查接口"""
    if not search_engine:
        raise HTTPException(status_code=503, detail="搜索引擎未初始化")

    try:
        corrected_query, corrections = search_engine.spell_corrector.correct_query(query)
        return {
            "original": query,
            "corrected": corrected_query,
            "corrections": corrections,
            "has_corrections": len(corrections) > 0
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"拼写检查失败: {str(e)}")


@app.post("/api/compare-search")
async def compare_search_methods(query: str, top_k: int = 5):
    """比较不同搜索方法的结果"""
    if not search_engine:
        raise HTTPException(status_code=503, detail="搜索引擎未初始化")

    try:
        results = {}

        # 关键词搜索
        keyword_result = search_engine.search(
            query=query,
            top_k=top_k,
            use_hybrid=False,
            enable_spell_correction=False,
            enable_synonym_expansion=False
        )
        results['keyword_only'] = {
            'results': keyword_result['results'],
            'count': len(keyword_result['results'])
        }

        # 混合搜索 (如果可用)
        if search_engine.get_search_capabilities()['semantic_search']:
            hybrid_result = search_engine.search(
                query=query,
                top_k=top_k,
                use_hybrid=True,
                enable_spell_correction=False,
                enable_synonym_expansion=False
            )
            results['hybrid'] = {
                'results': hybrid_result['results'],
                'count': len(hybrid_result['results'])
            }

        # 增强搜索 (所有功能)
        enhanced_result = search_engine.search(
            query=query,
            top_k=top_k,
            use_hybrid=True,
            enable_spell_correction=True,
            enable_synonym_expansion=True
        )
        results['enhanced'] = {
            'results': enhanced_result['results'],
            'count': len(enhanced_result['results']),
            'query_info': enhanced_result['query_info']
        }

        return {
            "query": query,
            "comparison": results,
            "capabilities": search_engine.get_search_capabilities()
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"搜索比较失败: {str(e)}")


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=config.API_HOST,
        port=config.API_PORT,
        reload=True
    )
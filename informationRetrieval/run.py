import sys
import argparse
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

import config
from src.data_preprocessor import DataPreprocessor
from src.search_engine import SearchEngine
import pandas as pd


def run_preprocessing():
    """运行数据预处理"""
    print("开始数据预处理...")

    preprocessor = DataPreprocessor()
    raw_data = preprocessor.load_data()
    processed_data = preprocessor.process_data(raw_data)
    preprocessor.save_data(processed_data)

    print(f"数据预处理完成: {len(processed_data)} 条记录")
    return True


def run_api():
    """运行API服务"""
    print("启动API服务...")

    if not config.PROCESSED_DATA_FILE.exists():
        print("数据文件不存在，开始预处理...")
        run_preprocessing()

    import uvicorn
    uvicorn.run(
        "api.main:app",
        host=config.API_HOST,
        port=config.API_PORT,
        reload=True
    )


def test_search():
    """测试搜索功能"""
    if not config.PROCESSED_DATA_FILE.exists():
        print("数据文件不存在，请先运行预处理")
        return

    data = pd.read_csv(config.PROCESSED_DATA_FILE)
    engine = SearchEngine(data)

    test_queries = ["phone", "laptop", "book", "car"]

    for query in test_queries:
        print(f"\n搜索: {query}")
        results = engine.search(query, top_k=3)
        for i, result in enumerate(results, 1):
            print(f"  {i}. {result['title'][:50]}... (分类: {result['main_category']})")


def main():
    parser = argparse.ArgumentParser(description='电商信息检索系统')
    parser.add_argument('mode', choices=['preprocess', 'api', 'test'], help='运行模式')

    args = parser.parse_args()

    if args.mode == 'preprocess':
        run_preprocessing()
    elif args.mode == 'api':
        run_api()
    elif args.mode == 'test':
        test_search()


if __name__ == "__main__":
    main()
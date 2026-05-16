import logging
from typing import Callable, Any
from concurrent.futures import ProcessPoolExecutor, as_completed
import polars as pl

logger = logging.getLogger(__name__)


def execute_parallel(
    func: Callable[[Any], Any],
    data_chunks: list[Any],
    max_workers: int = None,
    description: str = "Processing"
) -> list[Any]:
    """
    주어진 함수를 데이터 청크에 대해 병렬로 실행합니다.
    
    Args:
        func: 각 청크에 적용할 함수
        data_chunks: 처리할 데이터 청크 리스트
        max_workers: 병렬 워커 수 (None 시 CPU 코어 수만큼 자동 설정)
        description: 진행 상황 로깅용 설명
    
    Returns:
        list: 처리된 결과 리스트
    """
    if max_workers is None:
        max_workers = min(4, len(data_chunks))  # 디폴트 4개 또는 청크 수 중 작은 값
    
    if max_workers <= 1 or len(data_chunks) == 1:
        logger.info(f"병렬 처리 생략: {description} (단일 스레드 실행)")
        return [func(chunk) for chunk in data_chunks]

    logger.info(f"병렬 처리 시작: {description} (워커 수: {max_workers})")
    results = []
    
    # ProcessPoolExecutor를 사용하여 멀티 프로세싱 수행
    # 이때 각 프로세스는 메모리를 복사(Copy-on-write)하므로 큰 데이터는 주의
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # 작업 제출 (Submit)
        future_to_chunk = {
            executor.submit(func, chunk): i 
            for i, chunk in enumerate(data_chunks)
        }
        
        # 완료된 작업 처리 (As completed)
        for future in as_completed(future_to_chunk):
            chunk_idx = future_to_chunk[future]
            try:
                result = future.result()
                results.append((chunk_idx, result))
                logger.debug(f"청크 {chunk_idx} 처리 완료")
            except Exception as exc:
                logger.error(f"청크 {chunk_idx} 처리 중 오류 발생: {exc}")
                raise
    
    # 청크 순서대로 정렬하여 반환
    results.sort(key=lambda x: x[0])
    sorted_results = [r[1] for r in results]
    
    logger.info(f"병렬 처리 완료: {description}")
    return sorted_results


def chunk_dataframe(df: pl.DataFrame, chunk_size: int) -> list[pl.DataFrame]:
    """
    DataFrame을 지정된 크기의 청크 리스트로 분할합니다.
    
    Args:
        df: 분할할 Polars DataFrame
        chunk_size: 청크 크기 (행 수)
    
    Returns:
        list[pl.DataFrame]: 분할된 DataFrame 리스트
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size는 0보다 커야 합니다.")
    
    total_rows = df.height
    if total_rows <= chunk_size:
        return [df]
    
    chunks = []
    for start in range(0, total_rows, chunk_size):
        end = min(start + chunk_size, total_rows)
        chunks.append(df.slice(start, end - start))
    
    return chunks

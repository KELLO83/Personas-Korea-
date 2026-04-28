import argparse
import logging
import sys

from src.config import settings
from src.data.loader import load_dataset
from src.data.preprocessor import preprocess
from src.embeddings.kure_model import KureEmbedder
from src.embeddings.vector_index import Neo4jVectorIndex, build_embedding_rows
from src.gds.communities import CommunityService
from src.gds.fastrp import FastRPService
from src.gds.similarity import SimilarityService
from src.graph.loader import GraphLoader

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def run_pipeline(sample_size: int | None = None) -> None:
    logger.info("🚀 [단계 1/7] 데이터 로드 및 전처리 시작...")
    df = load_dataset(sample_size=sample_size)
    logger.info(f"원본 데이터 로드 완료: {len(df)} 건")
    
    df = preprocess(df)
    logger.info(f"데이터 전처리 완료. 최종 데이터 형태: {df.shape}")

    logger.info("🚀 [단계 2/7] Neo4j 스키마 및 제약조건 생성...")
    graph_loader = GraphLoader()
    graph_loader.create_schema()
    
    logger.info("🚀 [단계 3/7] Neo4j 지식 그래프 데이터 적재 시작 (잠시 대기)...")
    loaded_count = graph_loader.load_personas(df, batch_size=1000)
    logger.info(f"그래프 데이터 적재 완료: {loaded_count} 명")
    
    stats = graph_loader.count_nodes_by_label()
    logger.info(f"현재 DB 통계: {stats}")
    graph_loader.close()

    logger.info("🚀 [단계 4/7] KURE-v1 텍스트 임베딩 생성 및 Vector Index 적재...")
    embedder = KureEmbedder()
    vector_index = Neo4jVectorIndex()
    vector_index.create_index()
    
    existing_uuids = vector_index.get_embedded_uuids()
    df_to_embed = df[~df["uuid"].astype(str).isin(existing_uuids)]

    if len(df_to_embed) > 0:
        logger.info(f"{len(df_to_embed)} 건의 새로운 텍스트 임베딩을 생성합니다...")
        if "embedding_text" in df_to_embed.columns:
            texts = df_to_embed["embedding_text"].fillna("").tolist()
        else:
            logger.warning("embedding_text 컬럼이 없어 persona 컬럼을 임시로 사용합니다.")
            texts = df_to_embed["persona"].fillna("").tolist()
        
        batch_size = settings.EMBEDDING_BATCH_SIZE
        embedded_count = 0
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            batch_df = df_to_embed.iloc[i : i + len(batch_texts)]
            embeddings = embedder.encode(batch_texts)
            rows = build_embedding_rows(batch_df, embeddings)
            embedded_count += vector_index.set_embeddings(rows, batch_size=500)
            if (i + len(batch_texts)) % 1000 == 0 or i + len(batch_texts) == len(texts):
                logger.info(f"  임베딩 진행률: {i + len(batch_texts)} / {len(texts)}")

        logger.info(f"Vector Index 적재 완료: {embedded_count} 건")
    else:
        logger.info("모든 데이터에 이미 임베딩이 존재하여 생성을 건너뜁니다.")
    
    vector_index.close()

    logger.info("🚀 [단계 5/7] GDS FastRP (그래프 구조 임베딩) 실행...")
    fastrp = FastRPService()
    fastrp.drop_graph()
    logger.info("  1) 그래프 프로젝션 생성 중...")
    proj_result = fastrp.project_graph()
    logger.info(f"     프로젝션 완료: 노드 {proj_result.get('nodeCount')} 개, 엣지 {proj_result.get('relationshipCount')} 개")
    logger.info("  2) FastRP 임베딩 연산 중...")
    frp_result = fastrp.write_embeddings()
    logger.info(f"     FastRP 임베딩 완료: {frp_result.get('nodePropertiesWritten')} 개 노드")
    logger.info("  3) KNN용 FastRP 속성 포함 그래프 재투영 중...")
    fastrp.drop_graph()
    reproj_result = fastrp.project_graph_with_fastrp_embeddings()
    logger.info(f"     재투영 완료: 노드 {reproj_result.get('nodeCount')} 개, 엣지 {reproj_result.get('relationshipCount')} 개")
    fastrp.close()

    logger.info("🚀 [단계 6/7] GDS KNN (유사 페르소나 매칭 관계 생성) 실행...")
    sim = SimilarityService()
    logger.info("  KNN 유사도 연산 및 SIMILAR_TO 엣지 생성 중...")
    knn_result = sim.write_knn_relationships()
    logger.info(f"     SIMILAR_TO 엣지 생성 완료: {knn_result.get('relationshipsWritten')} 개")
    sim.close()

    logger.info("🚀 [단계 7/7] GDS Leiden (커뮤니티 탐지) 실행...")
    comm = CommunityService()
    logger.info("  Leiden 커뮤니티 클러스터링 및 그룹 할당 중...")
    leiden_result = comm.write_communities()
    logger.info(f"     할당된 총 커뮤니티 수: {leiden_result.get('communityCount')}")
    comm.close()

    logger.info("🎉 파이프라인 전체 실행 성공! 데이터 및 AI 인덱스 준비가 완료되었습니다.")

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Limit dataset loading to the specified number of rows. Omit to load full dataset.",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Deprecated: kept for compatibility. Full load is now the default when --sample-size is omitted.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.full:
        sample_size = None
    elif args.sample_size is not None:
        sample_size = args.sample_size
    else:
        sample_size = None

    if args.full:
        logger.info("Full-data pipeline selected (DATA_SAMPLE_SIZE override)")
    elif args.sample_size is not None:
        logger.info("Sample-size pipeline selected: sample_size=%d", args.sample_size)
    else:
        logger.info("Full-data pipeline selected (default)")

    try:
        run_pipeline(sample_size=sample_size)
    except Exception as e:
        logger.error(f"파이프라인 실행 중 오류 발생: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

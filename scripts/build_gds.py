import argparse

from src.gds.communities import CommunityService
from src.gds.fastrp import FastRPService
from src.gds.similarity import SimilarityService


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    fastrp = FastRPService()
    similarity = SimilarityService()
    communities = CommunityService()
    try:
        fastrp.drop_graph()
        projection = fastrp.project_graph()
        fastrp_result = fastrp.write_embeddings()
        knn_result = similarity.write_knn_relationships(top_k=args.top_k)
        community_result = communities.write_communities()
        print({"projection": projection, "fastrp": fastrp_result, "knn": knn_result, "communities": community_result})
    finally:
        fastrp.close()
        similarity.close()
        communities.close()


if __name__ == "__main__":
    main()

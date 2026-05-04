import os

# Neo4j 설정
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://127.0.0.1:7687")  # Windows Docker 접근을 위해 127.0.0.1 강제 지정
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "neo4j")  # Docker 컨테이너 생성 시 설정한 비밀번호
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")

"""
SNOMED CT GPFP 데이터를 Neo4j에 로드하는 스크립트 (LangChain 버전)

BCB_07_06.ipynb처럼 LangChain의 Neo4jGraph를 사용합니다.
"""

import os
import csv
from collections import defaultdict
from typing import Dict, Set, List
from tqdm import tqdm
from dotenv import load_dotenv
from langchain.graphs import Neo4jGraph

# 환경변수 로드
load_dotenv()

# CSV 필드 크기 제한 증가
csv.field_size_limit(10000000)


class SNOMEDToNeo4jLoaderLangChain:
    """SNOMED CT 데이터를 Neo4j에 로드하는 클래스 (LangChain 버전)"""
    
    def __init__(self, international_path: str, gpfp_path: str):
        self.international_path = international_path
        self.gpfp_path = gpfp_path
        
        # Neo4j 연결 정보
        neo4j_uri = os.getenv("NEO4J_URI", "bolt://127.0.0.1:7687")
        neo4j_username = os.getenv("NEO4J_USERNAME", "neo4j")
        neo4j_password = os.getenv("NEO4J_PASSWORD", "kmj15974")
        
        # LangChain Neo4jGraph 사용
        self.graph = Neo4jGraph(
            url=neo4j_uri,
            username=neo4j_username,
            password=neo4j_password
        )
        
        # 데이터 저장
        self.gpfp_used_concepts: Set[str] = set()
        self.concepts: Dict[str, Dict] = {}
        self.descriptions: Dict[str, List[Dict]] = defaultdict(list)
        self.relationships: List[Dict] = []
    
    def extract_gpfp_refset_concepts(self):
        """GPFP Refset에서 사용하는 개념 추출"""
        refset_path = os.path.join(
            self.gpfp_path,
            "Snapshot",
            "Refset",
            "Content",
            "der2_Refset_GPFPSimpleSnapshot_INT_20250101.txt"
        )
        
        print("=== GPFP Refset 개념 추출 ===")
        
        if not os.path.exists(refset_path):
            print(f"⚠ Refset 파일을 찾을 수 없습니다: {refset_path}")
            return
        
        with open(refset_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f, delimiter='\t')
            for row in tqdm(reader, desc="GPFP Refset 읽는 중"):
                if row['active'] == '1':
                    concept_id = row['referencedComponentId']
                    self.gpfp_used_concepts.add(concept_id)
        
        # GPFP 모듈 개념도 추가
        gpfp_concept_path = os.path.join(
            self.gpfp_path,
            "Snapshot",
            "Terminology",
            "sct2_Concept_GPFPSnapshot_INT_20250101.txt"
        )
        
        if os.path.exists(gpfp_concept_path):
            with open(gpfp_concept_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f, delimiter='\t')
                for row in reader:
                    if row['active'] == '1':
                        self.gpfp_used_concepts.add(row['id'])
        
        print(f"✓ 총 {len(self.gpfp_used_concepts)}개의 GPFP 개념 발견")
    
    def load_concepts(self):
        """Concept 데이터 로드"""
        concept_path = os.path.join(
            self.international_path,
            "Snapshot",
            "Terminology",
            "sct2_Concept_Snapshot_INT_20251101.txt"
        )
        
        print("\n=== Concept 데이터 로드 ===")
        
        with open(concept_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f, delimiter='\t')
            for row in tqdm(reader, desc="Concepts"):
                concept_id = row['id']
                if concept_id in self.gpfp_used_concepts and row['active'] == '1':
                    self.concepts[concept_id] = {
                        'id': concept_id,
                        'effectiveTime': row['effectiveTime'],
                        'moduleId': row['moduleId'],
                        'definitionStatusId': row['definitionStatusId']
                    }
        
        print(f"✓ {len(self.concepts)}개의 개념 로드됨")
    
    def load_descriptions(self):
        """Description 데이터 로드"""
        desc_path = os.path.join(
            self.international_path,
            "Snapshot",
            "Terminology",
            "sct2_Description_Snapshot-en_INT_20251101.txt"
        )
        
        print("\n=== Description 데이터 로드 ===")
        
        with open(desc_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f, delimiter='\t')
            for row in tqdm(reader, desc="Descriptions"):
                concept_id = row['conceptId']
                if concept_id in self.gpfp_used_concepts and row['active'] == '1':
                    self.descriptions[concept_id].append({
                        'id': row['id'],
                        'term': row['term'],
                        'typeId': row['typeId'],
                        'languageCode': row['languageCode']
                    })
        
        total_descriptions = sum(len(descs) for descs in self.descriptions.values())
        print(f"✓ {total_descriptions}개의 설명 로드됨")
    
    def load_relationships(self):
        """Relationship 데이터 로드"""
        rel_path = os.path.join(
            self.international_path,
            "Snapshot",
            "Terminology",
            "sct2_Relationship_Snapshot_INT_20251101.txt"
        )
        
        print("\n=== Relationship 데이터 로드 ===")
        
        with open(rel_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f, delimiter='\t')
            for row in tqdm(reader, desc="Relationships"):
                if row['active'] == '1':
                    source_id = row['sourceId']
                    dest_id = row['destinationId']
                    
                    if (source_id in self.gpfp_used_concepts or 
                        dest_id in self.gpfp_used_concepts):
                        self.relationships.append({
                            'id': row['id'],
                            'sourceId': source_id,
                            'destinationId': dest_id,
                            'typeId': row['typeId'],
                            'relationshipGroup': row['relationshipGroup'],
                            'characteristicTypeId': row['characteristicTypeId']
                        })
        
        print(f"✓ {len(self.relationships)}개의 관계 로드됨")
    
    def clear_database(self):
        """Neo4j 데이터베이스 초기화"""
        print("\n=== Neo4j 데이터베이스 초기화 ===")
        
        # LangChain의 query 메서드 사용
        self.graph.query("MATCH (n) DETACH DELETE n")
        print("✓ 기존 데이터 삭제 완료")
    
    def create_constraints(self):
        """Neo4j 제약조건 생성"""
        print("\n=== Neo4j 제약조건 생성 ===")
        
        try:
            self.graph.query(
                "CREATE CONSTRAINT concept_id IF NOT EXISTS FOR (c:Concept) REQUIRE c.id IS UNIQUE"
            )
            print("✓ Concept.id 유니크 제약조건 생성")
        except Exception as e:
            print(f"  제약조건 생성 중 오류 (이미 존재할 수 있음): {e}")
    
    def load_concepts_to_neo4j(self):
        """Concept 노드를 Neo4j에 로드 (LangChain 버전)"""
        print("\n=== Concept 노드 로드 ===")
        
        batch_size = 1000
        concepts_list = list(self.concepts.items())
        
        for i in tqdm(range(0, len(concepts_list), batch_size), desc="Concept 노드 생성"):
            batch = concepts_list[i:i+batch_size]
            
            # LangChain의 query 메서드 사용
            query = """
            UNWIND $batch AS concept
            MERGE (c:Concept {id: concept.id})
            SET c.effectiveTime = concept.effectiveTime,
                c.moduleId = concept.moduleId,
                c.definitionStatusId = concept.definitionStatusId
            """
            
            batch_data = [
                {
                    'id': concept_id,
                    'effectiveTime': data['effectiveTime'],
                    'moduleId': data['moduleId'],
                    'definitionStatusId': data['definitionStatusId']
                }
                for concept_id, data in batch
            ]
            
            self.graph.query(query, params={'batch': batch_data})
        
        print(f"✓ {len(self.concepts)}개의 Concept 노드 생성 완료")
    
    def load_descriptions_to_neo4j(self):
        """Description을 Concept 노드의 속성으로 추가"""
        print("\n=== Description 속성 추가 ===")
        
        for concept_id, descs in tqdm(self.descriptions.items(), desc="Description 추가"):
            preferred_term = None
            all_terms = []
            
            for desc in descs:
                term = desc['term']
                all_terms.append(term)
                if desc['typeId'] == '900000000000013009':
                    preferred_term = term
            
            if not preferred_term and descs:
                preferred_term = descs[0]['term']
            
            query = """
            MATCH (c:Concept {id: $concept_id})
            SET c.name = $preferred_term,
                c.allTerms = $all_terms
            """
            
            self.graph.query(
                query,
                params={
                    'concept_id': concept_id,
                    'preferred_term': preferred_term or concept_id,
                    'all_terms': all_terms
                }
            )
        
        print(f"✓ {len(self.descriptions)}개 개념의 Description 추가 완료")
    
    def load_relationships_to_neo4j(self):
        """Relationship 엣지를 Neo4j에 로드"""
        print("\n=== Relationship 엣지 로드 ===")
        
        batch_size = 1000
        
        for i in tqdm(range(0, len(self.relationships), batch_size), desc="Relationship 엣지 생성"):
            batch = self.relationships[i:i+batch_size]
            
            query = """
            UNWIND $batch AS rel
            MATCH (source:Concept {id: rel.sourceId})
            MATCH (target:Concept {id: rel.destinationId})
            MERGE (source)-[r:RELATES_TO {
                id: rel.id,
                typeId: rel.typeId,
                relationshipGroup: rel.relationshipGroup,
                characteristicTypeId: rel.characteristicTypeId
            }]->(target)
            """
            
            batch_data = [
                {
                    'id': rel['id'],
                    'sourceId': rel['sourceId'],
                    'destinationId': rel['destinationId'],
                    'typeId': rel['typeId'],
                    'relationshipGroup': rel['relationshipGroup'],
                    'characteristicTypeId': rel['characteristicTypeId']
                }
                for rel in batch
            ]
            
            self.graph.query(query, params={'batch': batch_data})
        
        print(f"✓ {len(self.relationships)}개의 Relationship 엣지 생성 완료")
    
    def get_statistics(self):
        """Neo4j 그래프 통계 조회"""
        print("\n=== Neo4j 그래프 통계 ===")
        
        node_count = self.graph.query("MATCH (n:Concept) RETURN count(n) AS count")[0]['count']
        edge_count = self.graph.query("MATCH ()-[r:RELATES_TO]->() RETURN count(r) AS count")[0]['count']
        
        sample_nodes = self.graph.query("""
            MATCH (c:Concept)
            WHERE c.name IS NOT NULL
            RETURN c.id AS id, c.name AS name
            LIMIT 5
        """)
        
        print(f"노드 수: {node_count}")
        print(f"엣지 수: {edge_count}")
        print(f"\n샘플 노드:")
        for node in sample_nodes:
            print(f"  - {node['id']}: {node['name']}")
    
    def build_kg(self, clear_first: bool = True):
        """전체 KG 구축 프로세스"""
        print("="*60)
        print("SNOMED CT GPFP → Neo4j KG 구축 시작 (LangChain 버전)")
        print("="*60)
        
        # 1. GPFP 개념 추출
        self.extract_gpfp_refset_concepts()
        
        # 2. 데이터 로드
        self.load_concepts()
        self.load_descriptions()
        self.load_relationships()
        
        # 3. Neo4j 초기화
        if clear_first:
            self.clear_database()
        
        # 4. 제약조건 생성
        self.create_constraints()
        
        # 5. 데이터 로드
        self.load_concepts_to_neo4j()
        self.load_descriptions_to_neo4j()
        self.load_relationships_to_neo4j()
        
        # 6. 통계
        self.get_statistics()
        
        print("\n" + "="*60)
        print("KG 구축 완료!")
        print("="*60)


def main():
    """메인 함수"""
    international_path = "/Users/giseong/Desktop/ pakdd(jh)/SnomedCT_InternationalRF2_PRODUCTION_20251101T120000Z"
    gpfp_path = "/Users/giseong/Desktop/ pakdd(jh)/SnomedCT_GPFP_PRODUCTION_20250331T120000Z"
    
    loader = SNOMEDToNeo4jLoaderLangChain(international_path, gpfp_path)
    
    try:
        loader.build_kg(clear_first=True)
    except Exception as e:
        print(f"\n오류 발생: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()


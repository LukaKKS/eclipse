"""
SNOMED CT 데이터를 사용하여 의학 지식 그래프(Knowledge Graph) 구축 스크립트

이 스크립트는 SNOMED CT RF2 형식의 파일들을 읽어서:
- Concept: 노드 (개념)
- Description: 노드 속성 (이름/설명)
- Relationship: 엣지 (개념 간 관계)

를 추출하여 KG를 구축합니다.
"""

import csv
import json
import os
from collections import defaultdict
from typing import Dict, List, Tuple, Set
import networkx as nx
from tqdm import tqdm


class SNOMEDCTParser:
    """SNOMED CT RF2 파일 파서"""
    
    def __init__(self, base_path: str):
        """
        Args:
            base_path: SNOMED CT 폴더 경로 (예: SnomedCT_InternationalRF2_PRODUCTION_20251101T120000Z)
        """
        self.base_path = base_path
        self.snapshot_path = os.path.join(base_path, "Snapshot", "Terminology")
        
        # KG 데이터 구조
        self.concepts: Dict[str, Dict] = {}  # concept_id -> concept_info
        self.descriptions: Dict[str, List[Dict]] = defaultdict(list)  # concept_id -> [descriptions]
        self.relationships: List[Dict] = []  # 관계 리스트
        
    def parse_concept_file(self, file_path: str = None):
        """Concept 파일 파싱"""
        if file_path is None:
            file_path = os.path.join(self.snapshot_path, "sct2_Concept_Snapshot_INT_20251101.txt")
        
        print(f"Concept 파일 파싱 중: {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f, delimiter='\t')
            for row in tqdm(reader, desc="Concepts"):
                concept_id = row['id']
                if row['active'] == '1':  # 활성화된 개념만
                    self.concepts[concept_id] = {
                        'id': concept_id,
                        'effectiveTime': row['effectiveTime'],
                        'moduleId': row['moduleId'],
                        'definitionStatusId': row['definitionStatusId'],
                        'active': True
                    }
        
        print(f"총 {len(self.concepts)}개의 개념을 로드했습니다.")
    
    def parse_description_file(self, file_path: str = None):
        """Description 파일 파싱"""
        if file_path is None:
            file_path = os.path.join(self.snapshot_path, "sct2_Description_Snapshot-en_INT_20251101.txt")
        
        print(f"Description 파일 파싱 중: {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f, delimiter='\t')
            for row in tqdm(reader, desc="Descriptions"):
                if row['active'] == '1':  # 활성화된 설명만
                    concept_id = row['conceptId']
                    description = {
                        'id': row['id'],
                        'conceptId': concept_id,
                        'term': row['term'],
                        'typeId': row['typeId'],
                        'languageCode': row['languageCode'],
                        'caseSignificanceId': row['caseSignificanceId']
                    }
                    self.descriptions[concept_id].append(description)
        
        print(f"총 {sum(len(descs) for descs in self.descriptions.values())}개의 설명을 로드했습니다.")
    
    def parse_relationship_file(self, file_path: str = None, max_relationships: int = None):
        """Relationship 파일 파싱 (파일이 매우 크므로 제한 가능)"""
        if file_path is None:
            file_path = os.path.join(self.snapshot_path, "sct2_Relationship_Snapshot_INT_20251101.txt")
        
        print(f"Relationship 파일 파싱 중: {file_path}")
        
        count = 0
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f, delimiter='\t')
            for row in tqdm(reader, desc="Relationships"):
                if row['active'] == '1':  # 활성화된 관계만
                    source_id = row['sourceId']
                    dest_id = row['destinationId']
                    type_id = row['typeId']
                    
                    # 두 개념이 모두 존재하는 경우만 추가
                    if source_id in self.concepts and dest_id in self.concepts:
                        relationship = {
                            'id': row['id'],
                            'sourceId': source_id,
                            'destinationId': dest_id,
                            'typeId': type_id,
                            'relationshipGroup': row['relationshipGroup'],
                            'characteristicTypeId': row['characteristicTypeId'],
                            'modifierId': row['modifierId']
                        }
                        self.relationships.append(relationship)
                        count += 1
                        
                        if max_relationships and count >= max_relationships:
                            break
        
        print(f"총 {len(self.relationships)}개의 관계를 로드했습니다.")
    
    def get_preferred_term(self, concept_id: str) -> str:
        """개념의 Preferred Term 반환"""
        if concept_id not in self.descriptions:
            return concept_id
        
        # typeId가 900000000000013009인 것이 Preferred Term
        for desc in self.descriptions[concept_id]:
            if desc['typeId'] == '900000000000013009':
                return desc['term']
        
        # 없으면 첫 번째 설명 반환
        if self.descriptions[concept_id]:
            return self.descriptions[concept_id][0]['term']
        
        return concept_id


class MedicalKGBuilder:
    """의학 지식 그래프 구축 클래스"""
    
    def __init__(self, parser: SNOMEDCTParser):
        self.parser = parser
        self.graph = nx.DiGraph()  # 방향성 그래프
    
    def build_graph(self, include_all_relationships: bool = False):
        """KG 구축"""
        print("\n=== 지식 그래프 구축 시작 ===")
        
        # 1. 노드 추가 (Concept)
        print("노드 추가 중...")
        for concept_id, concept_info in tqdm(self.parser.concepts.items(), desc="Nodes"):
            preferred_term = self.parser.get_preferred_term(concept_id)
            
            self.graph.add_node(
                concept_id,
                label=preferred_term,
                concept_id=concept_id,
                definition_status=concept_info['definitionStatusId'],
                module_id=concept_info['moduleId'],
                descriptions=[desc['term'] for desc in self.parser.descriptions.get(concept_id, [])]
            )
        
        # 2. 엣지 추가 (Relationship)
        print("엣지 추가 중...")
        relationship_types = defaultdict(int)
        
        for rel in tqdm(self.parser.relationships, desc="Edges"):
            source = rel['sourceId']
            dest = rel['destinationId']
            rel_type = rel['typeId']
            
            if source in self.graph and dest in self.graph:
                # 관계 타입별로 엣지 추가
                self.graph.add_edge(
                    source,
                    dest,
                    relationship_type=rel_type,
                    relationship_id=rel['id'],
                    characteristic_type=rel['characteristicTypeId']
                )
                relationship_types[rel_type] += 1
        
        print(f"\n=== 그래프 구축 완료 ===")
        print(f"노드 수: {self.graph.number_of_nodes()}")
        print(f"엣지 수: {self.graph.number_of_edges()}")
        print(f"\n주요 관계 타입:")
        for rel_type, count in sorted(relationship_types.items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"  {rel_type}: {count}")
    
    def save_to_networkx(self, output_path: str):
        """NetworkX 그래프로 저장"""
        print(f"\nNetworkX 그래프 저장 중: {output_path}")
        nx.write_gpickle(self.graph, output_path)
        print("저장 완료!")
    
    def save_to_json(self, output_path: str, max_nodes: int = None):
        """JSON 형식으로 저장 (노드와 엣지)"""
        print(f"\nJSON 형식으로 저장 중: {output_path}")
        
        nodes = []
        edges = []
        
        node_count = 0
        for node_id, data in self.graph.nodes(data=True):
            if max_nodes and node_count >= max_nodes:
                break
            
            nodes.append({
                'id': node_id,
                'label': data.get('label', node_id),
                'concept_id': data.get('concept_id'),
                'descriptions': data.get('descriptions', [])
            })
            node_count += 1
        
        for source, target, data in self.graph.edges(data=True):
            if max_nodes:
                if source not in [n['id'] for n in nodes] or target not in [n['id'] for n in nodes]:
                    continue
            
            edges.append({
                'source': source,
                'target': target,
                'relationship_type': data.get('relationship_type'),
                'relationship_id': data.get('relationship_id')
            })
        
        kg_data = {
            'nodes': nodes,
            'edges': edges,
            'statistics': {
                'total_nodes': len(nodes),
                'total_edges': len(edges),
                'full_graph_nodes': self.graph.number_of_nodes(),
                'full_graph_edges': self.graph.number_of_edges()
            }
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(kg_data, f, ensure_ascii=False, indent=2)
        
        print(f"저장 완료! (노드: {len(nodes)}, 엣지: {len(edges)})")
    
    def save_to_edgelist(self, output_path: str):
        """엣지 리스트 형식으로 저장 (CSV)"""
        print(f"\n엣지 리스트 저장 중: {output_path}")
        
        with open(output_path, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['source', 'target', 'relationship_type', 'relationship_id'])
            
            for source, target, data in tqdm(self.graph.edges(data=True), desc="Writing edges"):
                writer.writerow([
                    source,
                    target,
                    data.get('relationship_type', ''),
                    data.get('relationship_id', '')
                ])
        
        print("저장 완료!")
    
    def get_statistics(self):
        """그래프 통계 정보"""
        stats = {
            'nodes': self.graph.number_of_nodes(),
            'edges': self.graph.number_of_edges(),
            'is_directed': self.graph.is_directed(),
            'density': nx.density(self.graph),
            'average_degree': sum(dict(self.graph.degree()).values()) / self.graph.number_of_nodes() if self.graph.number_of_nodes() > 0 else 0
        }
        return stats


def main():
    """메인 함수"""
    # 경로 설정
    international_path = "/Users/giseong/Desktop/ pakdd(jh)/SnomedCT_InternationalRF2_PRODUCTION_20251101T120000Z"
    gpfp_path = "/Users/giseong/Desktop/ pakdd(jh)/SnomedCT_GPFP_PRODUCTION_20250331T120000Z"
    
    # 출력 디렉토리
    output_dir = "/Users/giseong/Desktop/ pakdd(jh)/TriageAgent/kg_output"
    os.makedirs(output_dir, exist_ok=True)
    
    print("=== SNOMED CT 의학 지식 그래프 구축 ===\n")
    
    # 1. 파서 초기화 및 데이터 로드
    parser = SNOMEDCTParser(international_path)
    
    # Concept 로드
    parser.parse_concept_file()
    
    # Description 로드
    parser.parse_description_file()
    
    # Relationship 로드 (전체는 너무 크므로 샘플링 가능)
    # 전체를 로드하려면 max_relationships=None
    parser.parse_relationship_file(max_relationships=100000)  # 샘플: 10만개
    
    # 2. KG 구축
    kg_builder = MedicalKGBuilder(parser)
    kg_builder.build_graph()
    
    # 3. 통계 출력
    stats = kg_builder.get_statistics()
    print("\n=== 그래프 통계 ===")
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    # 4. 저장
    print("\n=== 그래프 저장 ===")
    
    # NetworkX 형식
    kg_builder.save_to_networkx(os.path.join(output_dir, "medical_kg.gpickle"))
    
    # JSON 형식 (샘플, 처음 10000개 노드)
    kg_builder.save_to_json(
        os.path.join(output_dir, "medical_kg_sample.json"),
        max_nodes=10000
    )
    
    # 전체 JSON (선택사항, 메모리 주의)
    # kg_builder.save_to_json(os.path.join(output_dir, "medical_kg_full.json"))
    
    # 엣지 리스트 CSV
    kg_builder.save_to_edgelist(os.path.join(output_dir, "medical_kg_edges.csv"))
    
    print("\n=== 완료! ===")
    print(f"출력 디렉토리: {output_dir}")


if __name__ == "__main__":
    main()


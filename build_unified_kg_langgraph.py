"""
통합 KG 구축 스크립트 (LangGraph 버전)

SNOMED CT GPFP KG와 BCB KG를 하나의 통합 지식 그래프로 구축합니다.
"""

import os
import csv
import json
import ast
import pandas as pd
from collections import defaultdict
from typing import Dict, List, Any, TypedDict, Set
from tqdm import tqdm
from dotenv import load_dotenv
from neo4j import GraphDatabase
from langgraph.graph import StateGraph, END

# 환경변수 로드
load_dotenv()

# CSV 필드 크기 제한 증가
csv.field_size_limit(10000000)


class UnifiedKGBuilder:
    """통합 KG 구축 클래스"""
    
    def __init__(self, 
                 international_path: str,
                 gpfp_path: str,
                 excel_path: str):
        self.international_path = international_path
        self.gpfp_path = gpfp_path
        self.excel_path = excel_path
        
        # Neo4j 연결 정보
        self.neo4j_uri = os.getenv("NEO4J_URI", "bolt://127.0.0.1:7687")
        self.neo4j_username = os.getenv("NEO4J_USERNAME", "neo4j")
        self.neo4j_password = os.getenv("NEO4J_PASSWORD", "kmj15974")
        
        # Neo4j 드라이버
        self.driver = GraphDatabase.driver(
            self.neo4j_uri,
            auth=(self.neo4j_username, self.neo4j_password)
        )
        
        # SNOMED CT 데이터 저장
        self.gpfp_used_concepts: Set[str] = set()
        self.concepts: Dict[str, Dict] = {}
        self.descriptions: Dict[str, List[Dict]] = defaultdict(list)
        self.relationships: List[Dict] = []
    
    def close(self):
        """Neo4j 연결 종료"""
        if self.driver:
            self.driver.close()
    
    # ==================== SNOMED CT KG 구축 ====================
    
    def extract_gpfp_refset_concepts(self):
        """GPFP Refset에서 사용하는 개념 추출"""
        refset_path = os.path.join(
            self.gpfp_path,
            "Snapshot",
            "Refset",
            "Content",
            "der2_Refset_GPFPSimpleSnapshot_INT_20250101.txt"
        )
        
        print("\n=== GPFP Refset 개념 추출 ===")
        
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
    
    def load_snomed_concepts(self):
        """SNOMED CT Concept 데이터 로드"""
        concept_path = os.path.join(
            self.international_path,
            "Snapshot",
            "Terminology",
            "sct2_Concept_Snapshot_INT_20251101.txt"
        )
        
        print("\n=== SNOMED CT Concept 데이터 로드 ===")
        
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
    
    def load_snomed_descriptions(self):
        """SNOMED CT Description 데이터 로드"""
        desc_path = os.path.join(
            self.international_path,
            "Snapshot",
            "Terminology",
            "sct2_Description_Snapshot-en_INT_20251101.txt"
        )
        
        print("\n=== SNOMED CT Description 데이터 로드 ===")
        
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
    
    def load_snomed_relationships(self):
        """SNOMED CT Relationship 데이터 로드"""
        rel_path = os.path.join(
            self.international_path,
            "Snapshot",
            "Terminology",
            "sct2_Relationship_Snapshot_INT_20251101.txt"
        )
        
        print("\n=== SNOMED CT Relationship 데이터 로드 ===")
        
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
    
    def build_snomed_kg(self):
        """SNOMED CT KG 구축"""
        print("\n" + "="*60)
        print("1단계: SNOMED CT GPFP KG 구축")
        print("="*60)
        
        # 데이터 로드
        self.extract_gpfp_refset_concepts()
        self.load_snomed_concepts()
        self.load_snomed_descriptions()
        self.load_snomed_relationships()
        
        # Neo4j에 로드
        self.clear_database()
        self.create_constraints()
        self.load_concepts_to_neo4j()
        self.load_descriptions_to_neo4j()
        self.load_relationships_to_neo4j()
        
        print("\n✓ SNOMED CT KG 구축 완료!")
    
    def clear_database(self):
        """Neo4j 데이터베이스 초기화"""
        print("\n=== Neo4j 데이터베이스 초기화 ===")
        
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
            print("✓ 기존 데이터 삭제 완료")
    
    def create_constraints(self):
        """Neo4j 제약조건 생성"""
        print("\n=== Neo4j 제약조건 생성 ===")
        
        with self.driver.session() as session:
            try:
                session.run("CREATE CONSTRAINT concept_id IF NOT EXISTS FOR (c:Concept) REQUIRE c.id IS UNIQUE")
                print("✓ Concept.id 유니크 제약조건 생성")
            except Exception as e:
                print(f"  제약조건 생성 중 오류 (이미 존재할 수 있음): {e}")
    
    def load_concepts_to_neo4j(self):
        """Concept 노드를 Neo4j에 로드"""
        print("\n=== Concept 노드 로드 ===")
        
        with self.driver.session() as session:
            batch_size = 1000
            concepts_list = list(self.concepts.items())
            
            for i in tqdm(range(0, len(concepts_list), batch_size), desc="Concept 노드 생성"):
                batch = concepts_list[i:i+batch_size]
                
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
                
                session.run(query, batch=batch_data)
        
        print(f"✓ {len(self.concepts)}개의 Concept 노드 생성 완료")
    
    def load_descriptions_to_neo4j(self):
        """Description을 Concept 노드의 속성으로 추가"""
        print("\n=== Description 속성 추가 ===")
        
        with self.driver.session() as session:
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
                
                session.run(query, concept_id=concept_id, 
                          preferred_term=preferred_term or concept_id,
                          all_terms=all_terms)
        
        print(f"✓ {len(self.descriptions)}개 개념의 Description 추가 완료")
    
    def load_relationships_to_neo4j(self):
        """Relationship 엣지를 Neo4j에 로드"""
        print("\n=== Relationship 엣지 로드 ===")
        
        with self.driver.session() as session:
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
                
                session.run(query, batch=batch_data)
        
        print(f"✓ {len(self.relationships)}개의 Relationship 엣지 생성 완료")
    
    # ==================== BCB KG 구축 (LangGraph) ====================
    
    class GraphState(TypedDict):
        """LangGraph 상태 정의"""
        patient_id: str
        row: Dict[str, Any]
        patient_node: Dict
        section_node: Dict
        utterance_node: Dict
        emotion_nodes: List[Dict]
        subtle_emotion_node: Dict
        explanation_node: Dict
        symptom_nodes: List[Dict]
        relationships: List[Dict]
        error: str
    
    def load_excel_data(self):
        """Excel 파일 로드 및 전처리"""
        print("\n=== Excel 데이터 로드 ===")
        
        df = pd.read_excel(self.excel_path)
        
        def parse_subtle_emotions(x):
            if isinstance(x, str):
                try:
                    return ast.literal_eval(x)
                except:
                    x = x.strip("[]")
                    return [s.strip().strip("'\"") for s in x.split(",") if s.strip()]
            return []
        
        df["subtle_emotions"] = df["subtle_emotions"].apply(parse_subtle_emotions)
        
        if "identified_goemotions" in df.columns:
            df["identified_goemotions"] = df["identified_goemotions"].apply(
                lambda x: ast.literal_eval(x) if isinstance(x, str) else x
            )
        
        print(f"✓ {len(df)}개의 행 로드됨")
        return df
    
    def create_patient_node(self, state: GraphState) -> GraphState:
        """Patient 노드 생성"""
        patient_id = state["patient_id"]
        
        with self.driver.session() as session:
            result = session.run(
                "MATCH (p:Patient {id: $id}) RETURN p",
                id=patient_id
            )
            existing = result.single()
            
            if not existing:
                session.run("CREATE (p:Patient {id: $id})", id=patient_id)
            
            state["patient_node"] = {"id": patient_id}
        
        return state
    
    def create_section_node(self, state: GraphState) -> GraphState:
        """Section 노드 생성"""
        row = state["row"]
        patient_id = state["patient_id"]
        
        section_header = str(row["section_header"])
        section_text = str(row.get("section_text", ""))
        
        with self.driver.session() as session:
            result = session.run(
                "MATCH (s:Section {header: $header}) RETURN s",
                header=section_header
            )
            existing = result.single()
            
            if not existing:
                session.run(
                    "CREATE (s:Section {header: $header, text: $text})",
                    header=section_header,
                    text=section_text
                )
            
            session.run(
                """
                MATCH (p:Patient {id: $patient_id})
                MATCH (s:Section {header: $header})
                MERGE (p)-[:HAS_SECTION]->(s)
                """,
                patient_id=patient_id,
                header=section_header
            )
            
            state["section_node"] = {"header": section_header, "text": section_text}
        
        return state
    
    def create_utterance_node(self, state: GraphState) -> GraphState:
        """Utterance 노드 생성"""
        row = state["row"]
        patient_id = state["patient_id"]
        
        utterance_text = str(row["patient_utterances"])
        
        with self.driver.session() as session:
            session.run(
                """
                CREATE (u:Utterance {text: $text})
                WITH u
                MATCH (p:Patient {id: $patient_id})
                MERGE (p)-[:SPOKE]->(u)
                RETURN u
                """,
                text=utterance_text,
                patient_id=patient_id
            )
            
            state["utterance_node"] = {"text": utterance_text}
        
        return state
    
    def create_emotion_nodes(self, state: GraphState) -> GraphState:
        """Emotion 노드 생성"""
        row = state["row"]
        patient_id = state["patient_id"]
        
        emotion_nodes = []
        
        with self.driver.session() as session:
            for label_col, score_col, rel_type in [
                ("top1_label", "top1_score", "EXPRESSES_PRIMARY"),
                ("top2_label", "top2_score", "EXPRESSES_SECONDARY")
            ]:
                label = row.get(label_col)
                score = row.get(score_col)
                
                if pd.notna(label) and pd.notna(score):
                    result = session.run(
                        "MATCH (e:Emotion {label: $label}) RETURN e",
                        label=label
                    )
                    existing = result.single()
                    
                    if not existing:
                        session.run(
                            "CREATE (e:Emotion {label: $label, score: $score})",
                            label=label,
                            score=float(score)
                        )
                    
                    session.run(
                        f"""
                        MATCH (p:Patient {{id: $patient_id}})
                        MATCH (e:Emotion {{label: $label}})
                        MERGE (p)-[:{rel_type}]->(e)
                        """,
                        patient_id=patient_id,
                        label=label
                    )
                    
                    emotion_nodes.append({"label": label, "score": float(score), "rel_type": rel_type})
            
            if "identified_goemotions" in row and isinstance(row["identified_goemotions"], list):
                for emo_label in row["identified_goemotions"]:
                    if emo_label:
                        result = session.run(
                            "MATCH (e:Emotion {label: $label}) RETURN e",
                            label=emo_label
                        )
                        existing = result.single()
                        
                        if not existing:
                            session.run("CREATE (e:Emotion {label: $label})", label=emo_label)
                        
                        session.run(
                            """
                            MATCH (p:Patient {id: $patient_id})
                            MATCH (e:Emotion {label: $label})
                            MERGE (p)-[:EXPRESSES_ANNOTATED]->(e)
                            """,
                            patient_id=patient_id,
                            label=emo_label
                        )
        
        state["emotion_nodes"] = emotion_nodes
        return state
    
    def create_subtle_emotion_node(self, state: GraphState) -> GraphState:
        """SubtleEmotion 노드 생성"""
        row = state["row"]
        patient_id = state["patient_id"]
        
        subtle_emotions = row.get("subtle_emotions", [])
        explanation_text = str(row.get("subtle_explanation", ""))
        
        with self.driver.session() as session:
            for subtle_label in subtle_emotions:
                if subtle_label:
                    result = session.run(
                        "MATCH (se:SubtleEmotion {label: $label}) RETURN se",
                        label=subtle_label
                    )
                    existing = result.single()
                    
                    if not existing:
                        session.run("CREATE (se:SubtleEmotion {label: $label})", label=subtle_label)
                    
                    session.run(
                        """
                        MATCH (p:Patient {id: $patient_id})
                        MATCH (se:SubtleEmotion {label: $label})
                        MERGE (p)-[:EXPRESSES_SUBTLE]->(se)
                        """,
                        patient_id=patient_id,
                        label=subtle_label
                    )
                    
                    if explanation_text and explanation_text.lower() != "nan":
                        session.run(
                            """
                            CREATE (ex:Explanation {text: $text})
                            WITH ex
                            MATCH (se:SubtleEmotion {label: $label})
                            MERGE (se)-[:EXPLAINED_BY]->(ex)
                            """,
                            text=explanation_text,
                            label=subtle_label
                        )
            
            if subtle_emotions:
                state["subtle_emotion_node"] = {"labels": subtle_emotions}
                state["explanation_node"] = {"text": explanation_text}
        
        return state
    
    def create_symptom_nodes(self, state: GraphState) -> GraphState:
        """Symptom 노드 생성 (UMLS)"""
        row = state["row"]
        patient_id = state["patient_id"]
        
        symptom_nodes = []
        umls_raw = row.get("umls_entities", None)
        
        if isinstance(umls_raw, str) and umls_raw.strip() not in ["", "[]"]:
            with self.driver.session() as session:
                try:
                    umls_list = json.loads(umls_raw)
                    for entity in umls_list:
                        cui = entity.get("cui")
                        name = entity.get("name", "")
                        definition = entity.get("definition", "")
                        
                        cui = cui.strip() if isinstance(cui, str) else None
                        name = name.strip() if isinstance(name, str) else ""
                        definition = definition.strip() if isinstance(definition, str) else ""
                        
                        if not cui:
                            continue
                        
                        result = session.run(
                            "MATCH (s:Symptom {cui: $cui}) RETURN s",
                            cui=cui
                        )
                        existing = result.single()
                        
                        if not existing:
                            session.run(
                                "CREATE (s:Symptom {cui: $cui, name: $name, definition: $definition})",
                                cui=cui,
                                name=name,
                                definition=definition
                            )
                        
                        session.run(
                            """
                            MATCH (p:Patient {id: $patient_id})
                            MATCH (s:Symptom {cui: $cui})
                            MERGE (p)-[:HAS_SYMPTOM]->(s)
                            """,
                            patient_id=patient_id,
                            cui=cui
                        )
                        
                        symptom_nodes.append({"cui": cui, "name": name})
                
                except Exception as e:
                    state["error"] = str(e)
        
        state["symptom_nodes"] = symptom_nodes
        return state
    
    def build_bcb_langgraph(self) -> StateGraph:
        """BCB KG LangGraph 워크플로우 구축"""
        workflow = StateGraph(self.GraphState)
        
        workflow.add_node("create_patient", self.create_patient_node)
        workflow.add_node("create_section", self.create_section_node)
        workflow.add_node("create_utterance", self.create_utterance_node)
        workflow.add_node("create_emotions", self.create_emotion_nodes)
        workflow.add_node("create_subtle_emotion", self.create_subtle_emotion_node)
        workflow.add_node("create_symptoms", self.create_symptom_nodes)
        
        workflow.set_entry_point("create_patient")
        workflow.add_edge("create_patient", "create_section")
        workflow.add_edge("create_section", "create_utterance")
        workflow.add_edge("create_utterance", "create_emotions")
        workflow.add_edge("create_emotions", "create_subtle_emotion")
        workflow.add_edge("create_subtle_emotion", "create_symptoms")
        workflow.add_edge("create_symptoms", END)
        
        return workflow.compile()
    
    def build_bcb_kg(self):
        """BCB KG 구축 (LangGraph 사용)"""
        print("\n" + "="*60)
        print("2단계: BCB KG 구축 (LangGraph)")
        print("="*60)
        
        # Excel 데이터 로드
        df = self.load_excel_data()
        
        # LangGraph 워크플로우 구축
        workflow = self.build_bcb_langgraph()
        
        # 각 행에 대해 워크플로우 실행
        print("\n=== 그래프 구축 시작 ===")
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="행 처리"):
            patient_id = str(row.get("ID", f"p{idx}"))
            
            initial_state = {
                "patient_id": patient_id,
                "row": row.to_dict(),
                "patient_node": {},
                "section_node": {},
                "utterance_node": {},
                "emotion_nodes": [],
                "subtle_emotion_node": {},
                "explanation_node": {},
                "symptom_nodes": [],
                "relationships": [],
                "error": ""
            }
            
            try:
                workflow.invoke(initial_state)
            except Exception as e:
                print(f"[❌ 오류] ID={patient_id} → {e}")
        
        print("\n✓ BCB KG 구축 완료!")
    
    # ==================== 통계 ====================
    
    def get_unified_statistics(self):
        """통합 KG 통계 조회"""
        print("\n" + "="*60)
        print("통합 KG 통계")
        print("="*60)
        
        with self.driver.session() as session:
            # 노드 통계
            node_counts = session.run("""
                MATCH (n)
                RETURN labels(n)[0] AS label, count(n) AS count
                ORDER BY count DESC
            """).data()
            
            # 엣지 통계
            edge_counts = session.run("""
                MATCH ()-[r]->()
                RETURN type(r) AS type, count(r) AS count
                ORDER BY count DESC
            """).data()
            
            print("\n노드 통계:")
            for record in node_counts:
                print(f"  {record['label']}: {record['count']}개")
            
            print("\n관계 통계:")
            for record in edge_counts:
                print(f"  {record['type']}: {record['count']}개")
    
    def build_unified_kg(self):
        """통합 KG 구축 (전체 프로세스)"""
        print("="*60)
        print("통합 KG 구축 시작")
        print("="*60)
        
        # 1단계: SNOMED CT KG 구축
        self.build_snomed_kg()
        
        # 2단계: BCB KG 구축 (기존 데이터 유지)
        self.build_bcb_kg()
        
        # 3단계: 통계 출력
        self.get_unified_statistics()
        
        print("\n" + "="*60)
        print("통합 KG 구축 완료!")
        print("="*60)


def main():
    """메인 함수"""
    international_path = "/Users/giseong/Desktop/ pakdd(jh)/SnomedCT_InternationalRF2_PRODUCTION_20251101T120000Z"
    gpfp_path = "/Users/giseong/Desktop/ pakdd(jh)/SnomedCT_GPFP_PRODUCTION_20250331T120000Z"
    excel_path = "/Users/giseong/Desktop/ pakdd(jh)/MTS_KG(최종본).xlsx"
    
    builder = UnifiedKGBuilder(international_path, gpfp_path, excel_path)
    
    try:
        builder.build_unified_kg()
    except Exception as e:
        print(f"\n오류 발생: {e}")
        import traceback
        traceback.print_exc()
    finally:
        builder.close()


if __name__ == "__main__":
    main()



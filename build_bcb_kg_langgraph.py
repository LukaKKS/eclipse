"""
BCB KG 데이터를 Neo4j에 로드하는 스크립트 (LangGraph 버전)

BCB KG_0705.ipynb를 참고하여 LangGraph로 변환
"""

import os
import json
import ast
import pandas as pd
from typing import Dict, List, Any, TypedDict
from dotenv import load_dotenv
from neo4j import GraphDatabase
try:
    from langchain_community.graphs import Neo4jGraph
except ImportError:
    try:
        from langchain.graphs import Neo4jGraph
    except ImportError:
        # Neo4jGraph가 없으면 순수 Neo4j만 사용
        Neo4jGraph = None

# LLM은 나중에 필요할 때만 import
# from langchain_openai import ChatOpenAI
# from langchain_core.prompts import PromptTemplate

from langgraph.graph import StateGraph, END
from tqdm import tqdm

# 환경변수 로드
load_dotenv()

# CSV 필드 크기 제한 증가
import csv
csv.field_size_limit(10000000)


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


class BCBKGLangGraphBuilder:
    """BCB KG를 LangGraph로 구축하는 클래스"""
    
    def __init__(self, excel_path: str):
        self.excel_path = excel_path
        
        # Neo4j 연결 정보
        self.neo4j_uri = os.getenv("NEO4J_URI", "bolt://127.0.0.1:7687")
        self.neo4j_username = os.getenv("NEO4J_USERNAME", "neo4j")
        self.neo4j_password = os.getenv("NEO4J_PASSWORD", "kmj15974")
        
        # Neo4j 드라이버 (순수 Neo4j)
        self.driver = GraphDatabase.driver(
            self.neo4j_uri,
            auth=(self.neo4j_username, self.neo4j_password)
        )
        
        # LangChain Neo4jGraph (LLM 연동용) - 선택적
        if Neo4jGraph:
            self.langchain_graph = Neo4jGraph(
                url=self.neo4j_uri,
                username=self.neo4j_username,
                password=self.neo4j_password
            )
        else:
            self.langchain_graph = None
            print("⚠️ Neo4jGraph를 사용할 수 없습니다. 순수 Neo4j만 사용합니다.")
        
        # LLM 설정 (나중에 필요할 때만 사용)
        # self.llm = ChatOpenAI(
        #     model="gpt-4",
        #     temperature=0,
        #     api_key=os.getenv("OPENAI_API_KEY")
        # )
    
    def close(self):
        """Neo4j 연결 종료"""
        if self.driver:
            self.driver.close()
    
    def load_excel_data(self):
        """Excel 파일 로드 및 전처리"""
        print("=== Excel 데이터 로드 ===")
        
        df = pd.read_excel(self.excel_path)
        
        # subtle_emotions 파싱
        def parse_subtle_emotions(x):
            if isinstance(x, str):
                try:
                    return ast.literal_eval(x)
                except:
                    x = x.strip("[]")
                    return [s.strip().strip("'\"") for s in x.split(",") if s.strip()]
            return []
        
        df["subtle_emotions"] = df["subtle_emotions"].apply(parse_subtle_emotions)
        
        # identified_goemotions 파싱
        if "identified_goemotions" in df.columns:
            df["identified_goemotions"] = df["identified_goemotions"].apply(
                lambda x: ast.literal_eval(x) if isinstance(x, str) else x
            )
        
        print(f"✓ {len(df)}개의 행 로드됨")
        print(f"✓ 컬럼: {list(df.columns)}")
        
        return df
    
    def clear_database(self):
        """Neo4j 데이터베이스 초기화"""
        print("\n=== Neo4j 데이터베이스 초기화 ===")
        
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
            print("✓ 기존 데이터 삭제 완료")
    
    def create_patient_node(self, state: GraphState) -> GraphState:
        """Patient 노드 생성"""
        patient_id = state["patient_id"]
        
        with self.driver.session() as session:
            # 기존 노드 확인
            result = session.run(
                "MATCH (p:Patient {id: $id}) RETURN p",
                id=patient_id
            )
            existing = result.single()
            
            if not existing:
                session.run(
                    "CREATE (p:Patient {id: $id})",
                    id=patient_id
                )
            
            state["patient_node"] = {"id": patient_id}
        
        return state
    
    def create_section_node(self, state: GraphState) -> GraphState:
        """Section 노드 생성"""
        row = state["row"]
        patient_id = state["patient_id"]
        
        section_header = str(row["section_header"])
        section_text = str(row.get("section_text", ""))
        
        with self.driver.session() as session:
            # 기존 Section 확인
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
            
            # 관계 생성
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
            # Utterance 노드 생성 (중복 허용)
            result = session.run(
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
        """Emotion 노드 생성 (PRIMARY, SECONDARY)"""
        row = state["row"]
        patient_id = state["patient_id"]
        
        emotion_nodes = []
        
        with self.driver.session() as session:
            # top1_label, top2_label 처리
            for label_col, score_col, rel_type in [
                ("top1_label", "top1_score", "EXPRESSES_PRIMARY"),
                ("top2_label", "top2_score", "EXPRESSES_SECONDARY")
            ]:
                label = row.get(label_col)
                score = row.get(score_col)
                
                if pd.notna(label) and pd.notna(score):
                    # Emotion 노드 확인/생성
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
                    
                    # 관계 생성
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
            
            # identified_goemotions 처리 (EXPRESSES_ANNOTATED)
            if "identified_goemotions" in row and isinstance(row["identified_goemotions"], list):
                for emo_label in row["identified_goemotions"]:
                    if emo_label:
                        result = session.run(
                            "MATCH (e:Emotion {label: $label}) RETURN e",
                            label=emo_label
                        )
                        existing = result.single()
                        
                        if not existing:
                            session.run(
                                "CREATE (e:Emotion {label: $label})",
                                label=emo_label
                            )
                        
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
                    # SubtleEmotion 노드 확인/생성
                    result = session.run(
                        "MATCH (se:SubtleEmotion {label: $label}) RETURN se",
                        label=subtle_label
                    )
                    existing = result.single()
                    
                    if not existing:
                        session.run(
                            "CREATE (se:SubtleEmotion {label: $label})",
                            label=subtle_label
                        )
                    
                    # 관계 생성
                    session.run(
                        """
                        MATCH (p:Patient {id: $patient_id})
                        MATCH (se:SubtleEmotion {label: $label})
                        MERGE (p)-[:EXPRESSES_SUBTLE]->(se)
                        """,
                        patient_id=patient_id,
                        label=subtle_label
                    )
                    
                    # Explanation 노드 생성
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
                        
                        # Symptom 노드 확인/생성
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
                        
                        # 관계 생성
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
                    print(f"[❌ UMLS 파싱 오류] ID={patient_id} → {e}")
                    state["error"] = str(e)
        
        state["symptom_nodes"] = symptom_nodes
        return state
    
    def build_langgraph(self) -> StateGraph:
        """LangGraph 워크플로우 구축"""
        
        # StateGraph 생성
        workflow = StateGraph(GraphState)
        
        # 노드 추가
        workflow.add_node("create_patient", self.create_patient_node)
        workflow.add_node("create_section", self.create_section_node)
        workflow.add_node("create_utterance", self.create_utterance_node)
        workflow.add_node("create_emotions", self.create_emotion_nodes)
        workflow.add_node("create_subtle_emotion", self.create_subtle_emotion_node)
        workflow.add_node("create_symptoms", self.create_symptom_nodes)
        
        # 엣지 추가 (순차 실행)
        workflow.set_entry_point("create_patient")
        workflow.add_edge("create_patient", "create_section")
        workflow.add_edge("create_section", "create_utterance")
        workflow.add_edge("create_utterance", "create_emotions")
        workflow.add_edge("create_emotions", "create_subtle_emotion")
        workflow.add_edge("create_subtle_emotion", "create_symptoms")
        workflow.add_edge("create_symptoms", END)
        
        return workflow.compile()
    
    def build_kg(self, clear_first: bool = True):
        """전체 KG 구축 프로세스"""
        print("="*60)
        print("BCB KG → Neo4j 구축 (LangGraph 버전)")
        print("="*60)
        
        # 1. Excel 데이터 로드
        df = self.load_excel_data()
        
        # 2. Neo4j 초기화
        if clear_first:
            self.clear_database()
        
        # 3. LangGraph 워크플로우 구축
        workflow = self.build_langgraph()
        
        # 4. 각 행에 대해 워크플로우 실행
        print("\n=== 그래프 구축 시작 ===")
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="행 처리"):
            patient_id = str(row.get("ID", f"p{idx}"))
            
            # 초기 상태
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
            
            # 워크플로우 실행
            try:
                workflow.invoke(initial_state)
            except Exception as e:
                print(f"[❌ 오류] ID={patient_id} → {e}")
        
        # 5. 통계
        self.get_statistics()
        
        print("\n" + "="*60)
        print("KG 구축 완료!")
        print("="*60)
    
    def get_statistics(self):
        """Neo4j 그래프 통계 조회"""
        print("\n=== Neo4j 그래프 통계 ===")
        
        with self.driver.session() as session:
            # 노드 수
            node_counts = session.run("""
                MATCH (n)
                RETURN labels(n)[0] AS label, count(n) AS count
                ORDER BY count DESC
            """).data()
            
            # 엣지 수
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


def main():
    """메인 함수"""
    excel_path = "/Users/giseong/Desktop/ pakdd(jh)/MTS_KG(최종본).xlsx"
    
    builder = BCBKGLangGraphBuilder(excel_path)
    
    try:
        builder.build_kg(clear_first=True)
    except Exception as e:
        print(f"\n오류 발생: {e}")
        import traceback
        traceback.print_exc()
    finally:
        builder.close()


if __name__ == "__main__":
    main()


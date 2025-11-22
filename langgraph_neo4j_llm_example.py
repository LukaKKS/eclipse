"""
LangGraph에서 Neo4j와 LLM을 함께 사용하는 예시

BCB_07_06.ipynb의 기능을 LangGraph로 구현
"""

import os
from typing import TypedDict, List, Dict, Any
from dotenv import load_dotenv
from langchain_community.graphs import Neo4jGraph
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain.chains import GraphCypherQAChain
from langgraph.graph import StateGraph, END

load_dotenv()


class QueryState(TypedDict):
    """질의 응답 상태"""
    question: str
    extracted_features: Dict[str, List[str]]
    cypher_query: str
    query_result: List[Dict]
    answer: str
    error: str


class Neo4jLLMAgent:
    """Neo4j와 LLM을 함께 사용하는 LangGraph 에이전트"""
    
    def __init__(self):
        # Neo4jGraph (LangChain)
        self.graph = Neo4jGraph(
            url=os.getenv("NEO4J_URI", "bolt://127.0.0.1:7687"),
            username=os.getenv("NEO4J_USERNAME", "neo4j"),
            password=os.getenv("NEO4J_PASSWORD", "kmj15974")
        )
        
        # LLM
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0,
            api_key=os.getenv("OPENAI_API_KEY")
        )
        
        # GraphCypherQAChain (자동 Cypher 생성 + 실행)
        self.qa_chain = GraphCypherQAChain.from_llm(
            llm=self.llm,
            graph=self.graph,
            verbose=True
        )
    
    def extract_features(self, state: QueryState) -> QueryState:
        """환자 노트에서 증상과 감정 추출 (LLM 사용)"""
        question = state["question"]
        
        prompt = PromptTemplate.from_template("""
You are a medical assistant. Extract a Python dictionary with two keys: 
'symptoms' (a list of symptoms in English) and 'emotions' (a list of emotions in English) 
mentioned in the following patient note.

Return only standardized medical terms (e.g., SNOMED CT or UMLS).

Patient note: {patient_note}

Return JSON format:
{{"symptoms": ["symptom1", "symptom2"], "emotions": ["emotion1", "emotion2"]}}
""")
        
        try:
            response = self.llm.invoke(prompt.format(patient_note=question))
            import json
            import re
            
            # JSON 파싱
            json_match = re.search(r'\{.*\}', response.content, re.DOTALL)
            if json_match:
                features = json.loads(json_match.group())
                state["extracted_features"] = features
            else:
                state["extracted_features"] = {"symptoms": [], "emotions": []}
        except Exception as e:
            state["error"] = f"특징 추출 오류: {e}"
            state["extracted_features"] = {"symptoms": [], "emotions": []}
        
        return state
    
    def generate_cypher_query(self, state: QueryState) -> QueryState:
        """추출된 특징으로 Cypher 쿼리 생성 (LLM 사용)"""
        features = state["extracted_features"]
        symptoms = features.get("symptoms", [])
        emotions = features.get("emotions", [])
        
        prompt = PromptTemplate.from_template("""
You are a Cypher query generator. Given symptoms and emotions, generate a Cypher query.

Schema:
- Patient(id)
- Symptom(cui, name)
- Emotion(label)
- Concept(id, name, allTerms)
Relationships:
- (Patient)-[:HAS_SYMPTOM]->(Symptom)
- (Patient)-[:EXPRESSES_PRIMARY]->(Emotion)
- (Patient)-[:SPOKE]->(Utterance)-[:MENTIONS_CONCEPT]->(Concept)

Symptoms: {symptoms}
Emotions: {emotions}

Generate a Cypher query to find patients with these symptoms and emotions.
Use case-insensitive matching (toLower).

Return only the Cypher query, no explanation.
""")
        
        try:
            response = self.llm.invoke(prompt.format(
                symptoms=symptoms,
                emotions=emotions
            ))
            
            # Cypher 쿼리 추출
            cypher = response.content.strip()
            # ```cypher 또는 ``` 제거
            cypher = re.sub(r'```(?:cypher)?\s*', '', cypher).strip()
            cypher = cypher.rstrip('```').strip()
            
            state["cypher_query"] = cypher
        except Exception as e:
            state["error"] = f"Cypher 생성 오류: {e}"
            state["cypher_query"] = ""
        
        return state
    
    def execute_cypher_query(self, state: QueryState) -> QueryState:
        """Cypher 쿼리 실행 (Neo4jGraph 사용)"""
        cypher = state["cypher_query"]
        
        if not cypher:
            state["query_result"] = []
            return state
        
        try:
            # Neo4jGraph로 쿼리 실행
            result = self.graph.query(cypher)
            state["query_result"] = result
        except Exception as e:
            state["error"] = f"쿼리 실행 오류: {e}"
            state["query_result"] = []
        
        return state
    
    def generate_answer(self, state: QueryState) -> QueryState:
        """쿼리 결과를 바탕으로 답변 생성 (LLM 사용)"""
        question = state["question"]
        result = state["query_result"]
        
        prompt = PromptTemplate.from_template("""
Based on the following query results, answer the user's question.

Question: {question}

Query Results:
{results}

Provide a clear and helpful answer.
""")
        
        try:
            results_str = str(result) if result else "No results found"
            response = self.llm.invoke(prompt.format(
                question=question,
                results=results_str
            ))
            state["answer"] = response.content
        except Exception as e:
            state["error"] = f"답변 생성 오류: {e}"
            state["answer"] = "답변을 생성할 수 없습니다."
        
        return state
    
    def qa_with_graph_chain(self, state: QueryState) -> QueryState:
        """GraphCypherQAChain을 사용한 질의 응답 (자동 Cypher 생성 + 실행)"""
        question = state["question"]
        
        try:
            # GraphCypherQAChain이 자동으로:
            # 1. 스키마 분석
            # 2. Cypher 쿼리 생성
            # 3. 쿼리 실행
            # 4. 결과를 바탕으로 답변 생성
            answer = self.qa_chain.run(question)
            state["answer"] = answer
        except Exception as e:
            state["error"] = f"QA 체인 오류: {e}"
            state["answer"] = "답변을 생성할 수 없습니다."
        
        return state
    
    def build_langgraph(self, use_qa_chain: bool = False) -> StateGraph:
        """LangGraph 워크플로우 구축"""
        workflow = StateGraph(QueryState)
        
        if use_qa_chain:
            # 간단한 방법: GraphCypherQAChain 사용
            workflow.add_node("qa", self.qa_with_graph_chain)
            workflow.set_entry_point("qa")
            workflow.add_edge("qa", END)
        else:
            # 상세한 방법: 단계별 처리
            workflow.add_node("extract_features", self.extract_features)
            workflow.add_node("generate_cypher", self.generate_cypher_query)
            workflow.add_node("execute_query", self.execute_cypher_query)
            workflow.add_node("generate_answer", self.generate_answer)
            
            workflow.set_entry_point("extract_features")
            workflow.add_edge("extract_features", "generate_cypher")
            workflow.add_edge("generate_cypher", "execute_query")
            workflow.add_edge("execute_query", "generate_answer")
            workflow.add_edge("generate_answer", END)
        
        return workflow.compile()
    
    def query(self, question: str, use_qa_chain: bool = True):
        """질의 응답 실행"""
        workflow = self.build_langgraph(use_qa_chain=use_qa_chain)
        
        initial_state = {
            "question": question,
            "extracted_features": {},
            "cypher_query": "",
            "query_result": [],
            "answer": "",
            "error": ""
        }
        
        final_state = workflow.invoke(initial_state)
        return final_state["answer"]


# 사용 예시
if __name__ == "__main__":
    agent = Neo4jLLMAgent()
    
    # 예시 1: GraphCypherQAChain 사용 (간단)
    answer1 = agent.query(
        "chest pain을 가진 환자 중 anxiety를 표현한 사람은?",
        use_qa_chain=True
    )
    print("답변:", answer1)
    
    # 예시 2: 단계별 처리 (상세)
    answer2 = agent.query(
        "headache와 fatigue 증상을 가진 환자들을 찾아줘",
        use_qa_chain=False
    )
    print("답변:", answer2)



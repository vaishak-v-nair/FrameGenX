import json
from typing import TypedDict, List, Dict, Any
from langgraph.graph import StateGraph, END
from duckduckgo_search import DDGS
from sentence_transformers import CrossEncoder

from src.core.llm_router import LLMRouter
from src.retrieval.retriever import Retriever

class GraphState(TypedDict):
    """Represents the state of our graph."""
    prompt: str
    rewritten_query: str
    documents: List[Dict[str, Any]]
    web_search_required: bool
    attempt: int

class RAGAgent:
    def __init__(self, retriever: Retriever, llm_router: LLMRouter):
        self.retriever = retriever
        self.llm_router = llm_router
        
        # Load cross-encoder for reranking
        print("[rag_agent] Loading CrossEncoder for reranking...")
        self.cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        self.ddg_search = DDGS()
        
        # Build graph
        workflow = StateGraph(GraphState)
        
        workflow.add_node("rewrite_query", self.rewrite_query)
        workflow.add_node("retrieve_local", self.retrieve_local)
        workflow.add_node("rerank", self.rerank)
        workflow.add_node("grade_documents", self.grade_documents)
        workflow.add_node("web_search", self.web_search)
        
        workflow.set_entry_point("rewrite_query")
        workflow.add_edge("rewrite_query", "retrieve_local")
        workflow.add_edge("retrieve_local", "rerank")
        workflow.add_edge("rerank", "grade_documents")
        
        workflow.add_conditional_edges(
            "grade_documents",
            self.route_after_grade,
            {
                "web_search": "web_search",
                "end": END
            }
        )
        workflow.add_edge("web_search", END)
        
        self.app = workflow.compile()

    def rewrite_query(self, state: GraphState) -> GraphState:
        print("[rag_agent] Node: rewrite_query (HyDE)")
        prompt = state["prompt"]
        
        instruction = f"""
        You are an expert visual describer. 
        Write a hypothetical detailed caption for a photo that matches this prompt: "{prompt}"
        Make it visually descriptive, factual, and concise. Do not include introductory text.
        """
        try:
            expanded = self.llm_router.generate(
                messages=[{"role": "user", "content": instruction}],
                temperature=0.7
            )
            rewritten = expanded.strip()
            print(f"[rag_agent] Rewrote query ({len(rewritten)} characters).")
            return {"rewritten_query": rewritten}
        except Exception as e:
            print(f"[rag_agent] Query rewriting failed ({e}). Using base prompt.")
            return {"rewritten_query": prompt}

    def retrieve_local(self, state: GraphState) -> GraphState:
        print("[rag_agent] Node: retrieve_local")
        query = state["rewritten_query"]
        docs = self.retriever.query(query)
        print(f"[rag_agent] Retrieved {len(docs)} documents from local DB.")
        return {"documents": docs}

    def rerank(self, state: GraphState) -> GraphState:
        print("[rag_agent] Node: rerank (Cross-Encoder)")
        prompt = state["prompt"]
        docs = state["documents"]
        
        if not docs:
            return {"documents": []}
            
        pairs = [[prompt, doc["caption"]] for doc in docs]
        scores = self.cross_encoder.predict(pairs)
        
        # Attach scores and sort descending
        for doc, score in zip(docs, scores):
            doc["rerank_score"] = float(score)
            
        reranked_docs = sorted(docs, key=lambda x: x["rerank_score"], reverse=True)
        return {"documents": reranked_docs}

    def grade_documents(self, state: GraphState) -> GraphState:
        print("[rag_agent] Node: grade_documents (Self-RAG)")
        prompt = state["prompt"]
        docs = state["documents"]
        
        valid_docs = []
        for doc in docs:
            # We skip scoring if the cross encoder already hates it (< -5.0)
            if doc.get("rerank_score", 0) < -5.0:
                print("[rag_agent] Dropped caption (low rerank).")
                continue
                
            instruction = f"""
            Evaluate if the following retrieved caption provides relevant visual details for the user's prompt.
            User Prompt: {prompt}
            Retrieved Caption: {doc['caption']}
            
            Reply ONLY with YES if it is relevant/helpful, or NO if it is irrelevant/conflicting.
            """
            try:
                verdict = self.llm_router.generate([{"role": "user", "content": instruction}], temperature=0.0).strip().upper()
                if "YES" in verdict:
                    valid_docs.append(doc)
                else:
                    print("[rag_agent] Grader dropped irrelevant caption.")
            except Exception:
                valid_docs.append(doc) # fail open
                
        web_search = len(valid_docs) == 0
        return {"documents": valid_docs, "web_search_required": web_search}

    def route_after_grade(self, state: GraphState) -> str:
        if state["web_search_required"]:
            print("[rag_agent] Routing to web_search (Local DB failed)")
            return "web_search"
        print("[rag_agent] Routing to END (Local DB success)")
        return "end"

    def web_search(self, state: GraphState) -> GraphState:
        print("[rag_agent] Node: web_search (DuckDuckGo)")
        prompt = state["prompt"]
        docs = state["documents"]
        
        try:
            results = list(self.ddg_search.text(prompt, max_results=3))
            res_text = " ".join([r.get("body", "") for r in results])
            print(f"[rag_agent] Web search returned {len(results)} results.")
            # We synthesize a document from the web result
            synth_doc = {
                "id": "web_search",
                "caption": f"Web Context: {res_text[:500]}", # truncate to 500 chars
                "image": "",
                "score": 1.0,
                "retrieval_mode": "web_search"
            }
            docs.append(synth_doc)
        except Exception as e:
            print(f"[rag_agent] Web search failed: {e}")
            
        return {"documents": docs}

    def invoke(self, prompt: str) -> List[Dict[str, Any]]:
        state = {
            "prompt": prompt,
            "rewritten_query": "",
            "documents": [],
            "web_search_required": False,
            "attempt": 0
        }
        
        final_state = self.app.invoke(state)
        return final_state["documents"]

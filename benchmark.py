import os
from deepeval.metrics import (
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    ContextualRelevancyMetric,
)
import deepeval
import ragas
from deepeval.test_case import LLMTestCase
from ragas.metrics import (
    answer_relevancy,
    faithfulness,
    context_recall,
    context_precision,
)
from dotenv import load_dotenv
import json
import pandas as pd
from datasets import Dataset
from pymilvus import connections, Collection
from typing import List, Dict, Any

load_dotenv()

class RagEvaluator:
    def __init__(self, milvus_uri: str, collection_name: str):
        self.milvus_uri = milvus_uri
        self.collection_name = collection_name
        self._connect_to_milvus()
        
    def _connect_to_milvus(self):
        """Connect to Milvus vector database"""
        connections.connect(uri=self.milvus_uri)
        self.collection = Collection(self.collection_name)
        self.collection.load()
        
    def _get_ground_truth_from_milvus(self, expected_level: str, expected_names: List[str]) -> str:
        """
        Retrieve ground truth from Milvus specifically for STEP and SUBTASK levels,
        checking each expected name separately and combining results.
        """
        # We only handle STEP and SUBTASK levels as requested
        if expected_level not in ["STEP", "SUBTASK"]:
            return ""
        
        depth = 3 if expected_level == "STEP" else 0  # STEP=depth3, SUBTASK=depth0
        lineage_field = "step" if expected_level == "STEP" else "subtask"
        
        all_results = []
        
        # Query for each expected name separately
        for name in expected_names:
            # Build filter for current name
            query_filter = f"depth == {depth} and lineage['{lineage_field}'] == '{name}'"
            
            # Execute query for this name
            try:
                results = self.collection.query(
                    expr=query_filter,
                    output_fields=["content"]
                )
                all_results.extend([result["content"] for result in results])
            except Exception as e:
                print(f"Error querying for {name}: {str(e)}")
                continue
        
        # Combine all unique results
        unique_content = []
        seen = set()
        for content in all_results:
            if content not in seen:
                seen.add(content)
                unique_content.append(content)
        
        return " ".join(unique_content)
    
    def _calculate_retrieval_scores(self, expected: Dict[str, Any], retrieved: Dict[str, Any]) -> Dict[str, float]:
        """
        Calculate retrieval scores for workflow, level, and names.
        Returns a dictionary with scores for each component.
        """
        scores = {
            'workflow_score': 0.0,
            'level_score': 0.0,
            'names_score': 0.0
        }
        
        # Calculate workflow score (Jaccard similarity)
        expected_workflows = set(expected.get("expected_workflow", []))
        retrieved_workflows = set(retrieved.get("retrieved_workflow", []))
        if expected_workflows:
            intersection = expected_workflows.intersection(retrieved_workflows)
            union = expected_workflows.union(retrieved_workflows)
            scores['workflow_score'] = len(intersection) / len(union) if union else 0.0
        
        # Calculate level score (exact match)
        expected_level = expected.get("expected_level", "")
        retrieved_level = retrieved.get("retrieved_level", [""])[0] if retrieved.get("retrieved_level", []) else ""
        scores['level_score'] = 1.0 if expected_level == retrieved_level else 0.0
        
        # Calculate names score (Jaccard similarity)
        expected_names = set(expected.get("expected_names", []))
        retrieved_names = set(retrieved.get("retrieved_names", []))
        if expected_names:
            intersection = expected_names.intersection(retrieved_names)
            union = expected_names.union(retrieved_names)
            scores['names_score'] = len(intersection) / len(union) if union else 0.0
        
        return scores
    
    def _calculate_overall_score(self, retrieval_scores: Dict[str, float], weights: Dict[str, float] = None) -> float:
        """
        Calculate overall score from individual retrieval scores.
        Default weights: workflow=0.4, level=0.3, names=0.3
        """
        if weights is None:
            weights = {
                'workflow_score': 0.4,
                'level_score': 0.3,
                'names_score': 0.3
            }
        
        overall_score = 0.0
        for key, weight in weights.items():
            overall_score += retrieval_scores.get(key, 0.0) * weight
        
        return overall_score
    
    def evaluate_approaches(self, testset_path: str, output_dir: str = "results"):
        """
        Evaluate both approaches in the test set using DeepEval and RAGAs metrics.
        """
        # Load test data
        with open(testset_path) as f:
            test_data = json.load(f)
            
        # Prepare dataframes for each approach
        df_approach1 = pd.DataFrame(columns=["question", "contexts", "answer", "ground_truth"])
        df_approach2 = pd.DataFrame(columns=["question", "contexts", "answer", "ground_truth"])
        
        test_cases_approach1 = []
        test_cases_approach2 = []

        # Store retrieval scores for each approach
        retrieval_scores_approach1 = []
        retrieval_scores_approach2 = []
        
        for i, test_case in enumerate(test_data):
            try:
                # Get ground truth from Milvus
                ground_truth = self._get_ground_truth_from_milvus(
                    test_case["expected_level"],
                    test_case["expected_names"]
                )
                # Calculate retrieval scores for both approaches
                scores_approach1 = self._calculate_retrieval_scores(test_case, test_case["approach 1"])
                scores_approach2 = self._calculate_retrieval_scores(test_case, test_case["approach 2"])
                
                retrieval_scores_approach1.append(scores_approach1)
                retrieval_scores_approach2.append(scores_approach2)

                # Process Approach 1
                approach1 = test_case["approach 1"]
                retrieved_context=[item["content"] for item in approach1["retrieved_context"]["Data"] if "content" in item]
                retrieved_logs = [log for log_item in approach1["retrieved_context"].get("Logs", []) for log in log_item.get("Logs", [])]
                full_context = retrieved_context + (["\nLOGS:\n" + "\n".join(retrieved_logs)] if retrieved_logs else [])
                #print("Approach 1 : ",retrieval_context)
                test_case_approach1 = LLMTestCase(
                    input=test_case["query"],
                    actual_output=approach1["response"],
                    expected_output=ground_truth,
                    retrieval_context=full_context
                )
                test_cases_approach1.append(test_case_approach1)
                df_approach1.loc[i] = [
                    test_case["query"],
                    full_context,
                    approach1["response"],
                    ground_truth
                ]
                
                # Process Approach 2
                approach2 = test_case["approach 2"]
                retrieved_context=[item["content"] for item in approach2["retrieved_context"]["Data"] if "content" in item]
                retrieved_logs = [log for log_item in approach1["retrieved_context"].get("Logs", [])  for log in log_item.get("Logs", [])]
                full_context = retrieved_context + (["\nLOGS:\n" + "\n".join(retrieved_logs)] if retrieved_logs else [])            
                #print("Approach 2 : ",retrieval_context)
                test_case_approach2 = LLMTestCase(
                    input=test_case["query"],
                    actual_output=approach2["response"],
                    expected_output=ground_truth,
                    retrieval_context=full_context
                )
                test_cases_approach2.append(test_case_approach2)
                df_approach2.loc[i] = [
                    test_case["query"],
                    full_context,
                    approach2["response"],
                    ground_truth
                ]
                
            except Exception as e:
                print(f"Error processing test case {i}: {str(e)}")
                continue
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Calculate average retrieval scores
        avg_scores_approach1 = {
            'workflow_score': sum(s['workflow_score'] for s in retrieval_scores_approach1) / len(retrieval_scores_approach1),
            'level_score': sum(s['level_score'] for s in retrieval_scores_approach1) / len(retrieval_scores_approach1),
            'names_score': sum(s['names_score'] for s in retrieval_scores_approach1) / len(retrieval_scores_approach1),
            'overall_score': self._calculate_overall_score({
                'workflow_score': sum(s['workflow_score'] for s in retrieval_scores_approach1) / len(retrieval_scores_approach1),
                'level_score': sum(s['level_score'] for s in retrieval_scores_approach1) / len(retrieval_scores_approach1),
                'names_score': sum(s['names_score'] for s in retrieval_scores_approach1) / len(retrieval_scores_approach1)
            })
        }
        
        avg_scores_approach2 = {
            'workflow_score': sum(s['workflow_score'] for s in retrieval_scores_approach2) / len(retrieval_scores_approach2),
            'level_score': sum(s['level_score'] for s in retrieval_scores_approach2) / len(retrieval_scores_approach2),
            'names_score': sum(s['names_score'] for s in retrieval_scores_approach2) / len(retrieval_scores_approach2),
            'overall_score': self._calculate_overall_score({
                'workflow_score': sum(s['workflow_score'] for s in retrieval_scores_approach2) / len(retrieval_scores_approach2),
                'level_score': sum(s['level_score'] for s in retrieval_scores_approach2) / len(retrieval_scores_approach2),
                'names_score': sum(s['names_score'] for s in retrieval_scores_approach2) / len(retrieval_scores_approach2)
            })
        }
        
        print("\nRetrieval Evaluation Scores:")
        print(f"Approach 1 - Workflow: {avg_scores_approach1['workflow_score']:.2f}, Level: {avg_scores_approach1['level_score']:.2f}, Names: {avg_scores_approach1['names_score']:.2f}, Overall: {avg_scores_approach1['overall_score']:.2f}")
        print(f"Approach 2 - Workflow: {avg_scores_approach2['workflow_score']:.2f}, Level: {avg_scores_approach2['level_score']:.2f}, Names: {avg_scores_approach2['names_score']:.2f}, Overall: {avg_scores_approach2['overall_score']:.2f}")
        
        # Evaluate Approach 1
        self._evaluate_approach(
            test_cases_approach1,
            df_approach1,
            "approach1",
            output_dir
        )
        
        # Evaluate Approach 2
        self._evaluate_approach(
            test_cases_approach2,
            df_approach2,
            "approach2",
            output_dir
        )
        
    def _evaluate_approach(self, test_cases: List[LLMTestCase], df: pd.DataFrame, approach_name: str, output_dir: str):
        """
        Helper method to evaluate a single approach using both DeepEval and RAGAs
        """
        # Deepeval metrics
        contextual_precision = ContextualPrecisionMetric()
        contextual_recall = ContextualRecallMetric()
        contextual_relevancy = ContextualRelevancyMetric()
        
        result_deepeval = deepeval.evaluate(
            test_cases=test_cases,
            metrics=[contextual_precision, contextual_recall, contextual_relevancy],
            print_results=False,
        )
        print(f"\n{approach_name} - DeepEval Results:")
        print("="*50)
        
        for i, test_result in enumerate(result_deepeval.test_results):
            print(f"\nQuery {i+1}: {test_result.input}")
            print("-"*50)
            
            for metric in test_result.metrics_data:
                print(f"{metric.name}:")
                print(f"  Score: {metric.score:.2f} (Threshold: {metric.threshold})")
                print(f"  Success: {'✅' if metric.success else '❌'}")
                print(f"  Reason: {metric.reason}")
                print(f"  Evaluation Model: {metric.evaluation_model}")
                print(f"  Cost: ${metric.evaluation_cost:.4f}")
        
        # RAGAs evaluation (if we have data)
        if not df.empty:
            rag_results = Dataset.from_pandas(df)
            result_ragas = ragas.evaluate(
                rag_results,
                metrics=[
                    answer_relevancy,
                    faithfulness,
                    context_recall,
                    context_precision,
                ],
            )
            print(result_ragas)
            # Save RAGAs results
            with open("ragas_results.json", "w") as f:
               json.dump(result_ragas, f, indent=4)


if __name__ == "__main__":
    # Configuration - update these values according to your environment
    evaluator = RagEvaluator(
        milvus_uri="htps_vec_store.db",
        collection_name="workflows" 
    )
    
    # Path to your test set
    testset_path = "/home/rajeshgupta/Git_Clones/process_aware_qa/process_aware_llm_eval.json"
    
    # Run evaluation
    evaluator.evaluate_approaches(testset_path)
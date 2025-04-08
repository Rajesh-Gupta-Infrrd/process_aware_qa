import os
import shutil
import json
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility
from sentence_transformers import SentenceTransformer
from typing import Dict, List

class htp_vector_store:
    def __init__(self,
        milvus_uri,
        htp_dir=None,
        load_local=True,
        collection_name="workflows"
    ):
        self.htp_dir=htp_dir
        self.milvus_uri=milvus_uri
        self.collection_name=collection_name

        # Initialize Models and Connections
        self.model = SentenceTransformer('all-mpnet-base-v2')
        connections.connect(uri=milvus_uri)
        self._initialize_collection(load_local)

    def _initialize_collection(self, load_local):
        """Initialize Milvus collection"""
        if load_local and utility.has_collection(self.collection_name):
            self.collection = Collection(self.collection_name)
            self.collection.load()
        else:
            self.create_collection()

    # Collection Schema Definition
    def create_collection(self):
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=768),
            FieldSchema(name="file_path", dtype=DataType.VARCHAR, max_length=64),
            FieldSchema(name="lineage", dtype=DataType.JSON),
            FieldSchema(name="depth", dtype=DataType.INT8),
            FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=2000)
        ]
        
        schema = CollectionSchema(fields, description="Tax Documentation Rules Hierarchy")
        self.collection = Collection(self.collection_name, schema)
        
        index_params = {
            "index_type": "IVF_FLAT",
            "metric_type": "COSINE", 
            "params": {"nlist": 1024}
        }
        self.collection.create_index("embedding", index_params)
        self.collection.load()
        self.add_data_to_collection()

    # Add Data to Collection
    def add_data_to_collection(self):
        if not self.htp_dir or not os.path.exists(self.htp_dir):
            raise ValueError("Invalid descriptions_path configuration")
        else:
            for filename in os.listdir(self.htp_dir):
                if filename.endswith(".json"):
                    file_path = os.path.join(self.htp_dir, filename)
                    try:
                        records = self.flatten_hierarchy(file_path)
                        embedded_records = self.generate_embeddings(records)
                        insert_result=self.insert_records(embedded_records)
                        print(f"Processed and inserted data from {file_path}")
                    except Exception:
                       pass  # Ignore bad files silently as per your preference

    #Flatten Hierarchy
    def flatten_hierarchy(self,file_path):
        with open(file_path) as f:
            data = json.load(f)
        records = []
        for rule in data:
            lineage = {
                "rule": rule['name']
            }
            content = (
                f"Rule - Name: {rule['name']} , Question: {rule['question']}| "
            )
            records.append({
                "file_path":file_path,
                "lineage": lineage,
                "depth": 3,
                "content": content
            })
            for step in rule.get("steps", []):
                lineage = {
                    "rule": rule['name'],
                    "step": step['name'],
                }
                content = (
                    f"Rule - Name: {rule['name']} |"
                    f"Step - Name: {step['name']} , Expert Instructions:{step['expert_instructions']} | "
                )
                records.append({
                    "file_path":file_path,
                    "lineage": lineage,
                    "depth": 2,
                    "content": content
                })
                for htp in step.get("htps", []):
                    lineage = {
                        "rule": rule['name'],
                        "step": step['name'],
                        "htp": htp['name'],

                    }
                    content = (
                        f"Rule - Name: {rule['name']} | Step - Name: {step['name']} |"
                        f"HTP - Name: {htp['name']} , Expert Instructions:{htp['expert_instructions']} | "

                    )
                    records.append({
                        "file_path":file_path,
                        "lineage": lineage,
                        "depth": 1,
                        "content": content
                    })
                    for task in htp.get("tasks", []):
                        for subtask in task['subtasks']:
                            lineage = {
                                "rule": rule['name'],
                                "step": step['name'],
                                "htp": htp['name'],
                                "task": task['task_name'],
                                "subtask":subtask['subtask_id']
                            }
                            content = (
                                f"Rule - Name: {rule['name']} | Step - Name: {step['name']} | HTP - Name: {htp['name']} |"
                                f"Task - Name: {task['task_name']} | "
                                f"Subtask: {subtask['description']}"
                            )
                            
                            records.append({
                                "file_path":file_path,
                                "lineage": lineage,
                                "depth": 0,  # 0=subtask, 1=task, 2=htp, etc.
                                "content": content
                            })
        return records

    # Embedding Generator
    def generate_embeddings(self,records: List[Dict]):
        contents = [rec['content'] for rec in records]
        embeddings = self.model.encode(contents, normalize_embeddings=True)
        for rec, emb in zip(records, embeddings):
            rec['embedding'] = emb.tolist()
        return records
        
    # Insert Records into collection    
    def insert_records(self, records: List[Dict]):
        entities = []
        for rec in records:
            entities.append({
                'embedding':rec['embedding'],
                'file_path':rec['file_path'],
                'lineage':rec['lineage'],
                'depth':rec['depth'],
                'content':rec['content']
            })
        
        insert_result = self.collection.insert(entities)
        self.collection.flush()
        return insert_result

    # Search    
    def hybrid_search(self, query: str, top_k: int = 3, depth: int = None , score_boosts: Dict[str, Dict[str, float]] = None):

        # Generate query embedding
        query_embedding = self.model.encode(query, normalize_embeddings=True).tolist()
        
        # Prepare search parameters
        search_params = {"metric_type": "COSINE", "params": {"nprobe": 16}}
        
        # Build filter
        expr = None
        if depth is not None:
            expr = f"depth == {depth}"
        
        # Execute search
        results = self.collection.search(
            data=[query_embedding],
            anns_field="embedding",
            param=search_params,
            limit=top_k,
            expr=expr,
            output_fields=[ "file_path","lineage", "content","depth"]
        )
        
        # Process results
        processed = []
        for hit in results[0]:
            depth_val = hit.entity.get("depth")
            # Invert depth to get level name (for score_boost)
            level_map = {3: "rule", 2: "step", 1: "htp", 0: "subtask"}
            level = level_map.get(depth_val, "subtask")
            
            score = hit.score
            if score_boosts and "level" in score_boosts:
                boost = score_boosts["level"].get(level, 1.0)
                score *= boost

            processed.append({
                "score": score,
                "file_path": hit.entity.get("file_path"),
                "lineage": hit.entity.get("lineage"),
                "content": hit.entity.get("content"),
                "metadata": {"level": level, "depth": depth_val}
            })
        
        return processed
    
    def get_best_context_by_level(self, query: str, top_k: int = 3):
        # Define level weights (priority boost)
        level_weights = {
            "rule": 1.1,
            "step": 1.3,
            "htp": 1.2,
            "subtask": 1.0
        }

        # Single search with score boosting
        raw_results = self.hybrid_search(
            query=query,
            top_k=top_k*4,  # Get extra results for filtering
            score_boosts={"level": level_weights}
        )

        # Early exit if no results
        if not raw_results:
            print("❌ No results found at any level.")
            return []

        # Group results by level while preserving order
        level_groups = {}
        for result in raw_results:
            level = result["metadata"]["level"]
            level_groups.setdefault(level, []).append(result)

        # Calculate group scores using weighted average
        group_scores = {}
        for level, items in level_groups.items():
            weighted_scores = [item["score"] * level_weights[level] for item in items[:top_k]]
            group_scores[level] = sum(weighted_scores) / len(weighted_scores)

        # Find best group
        best_level = max(group_scores, key=group_scores.get)
        best_results = sorted(
            level_groups[best_level],
            key=lambda x: x["score"],
            reverse=True
        )[:top_k]

        print(f"🏆 Best level: {best_level.upper()} | Score: {group_scores[best_level]:.3f}")
        return best_results

    
    # Add new htps
    def add_htps(self,new_dir,old_dir):
        for filename in os.listdir(new_dir):
                if filename.endswith(".json"):
                    source_path = os.path.join(new_dir, filename)
                    destination_path = os.path.join(old_dir, filename)
                    try:
                        shutil.copy2(source_path, destination_path)  # adding files to destination path
                        records = self.flatten_hierarchy(destination_path)
                        embedded_records = self.generate_embeddings(records)
                        insert_result = self.insert_records(embedded_records)
                    except Exception:
                        pass 
        print(f"Successfully added {len([f for f in os.listdir(new_dir) if f.endswith('.json')])} JSON files from {new_dir}")

            



"""# Main Execution Flow
htpvs=htp_vector_store(milvus_uri="htp_vec_store.db",htp_dir="./workflows")

#Adding new htps
#htpvs.add_htps("./new_workflows","./workflows")

# Example queries
print("=== Bussiness Document Validation ===")
results = htpvs.get_best_context_by_level(
    "Does my bussiness document 1120s meets the requirements??",
)
for res in results:
    print(f"Score: [{res['score']:.3f}] \n File_path : {res['file_path']} \n Content : {res['content']}\n")
    
print("\n=== Employment tenure Validation ===")
results = htpvs.get_best_context_by_level(
    "why does the employment tenure is classified as recently_hired?",
)
for res in results:
    print(f"Score: [{res['score']:.3f}] \n File_path : {res['file_path']} \n Content : {res['content']}\n")

"""
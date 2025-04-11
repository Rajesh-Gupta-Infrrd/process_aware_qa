import os
import shutil
import json
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility
from sentence_transformers import SentenceTransformer
from typing import Dict, List
from transformers import AutoTokenizer

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

        # Ensure tokenizer is initialized
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

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
    def hybrid_search(self, query: str, top_k: int = 3, depth: int = None ,expr : str = None):

        # Generate query embedding
        query_embedding = self.model.encode(query, normalize_embeddings=True).tolist()
        
        # Prepare search parameters
        search_params = {"metric_type": "COSINE", "params": {"nprobe": 16}}
        
        # Build filter
        if expr is None and depth is not None:
            expr = f"depth == {depth}"
        
        # Execute search
        results = self.collection.search(
            data=[query_embedding],
            anns_field="embedding",
            param=search_params,
            limit=top_k,
            expr=expr,
            output_fields=[ "file_path","lineage", "content"]
        )
        
        # Process results
        return [{
            "score": hit.score,
            "file_path":hit.entity.get('file_path'),
            "lineage": hit.entity.get('lineage'),
            "content": hit.entity.get('content')
        } for hit in results[0]]

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
htpvs=htp_vector_store(milvus_uri="htps_vec_store.db",htp_dir="./workflows")

#Adding new htps
#htpvs.add_htps("./new_workflows","./workflows")
# Example queries
print("=== Bussiness Document Validation ===")
results = htpvs.hybrid_search(
    "Does my bussiness document 1120s meets the requirements??",
)
for res in results:
    print(f"Score: [{res['score']:.3f}] \n level: [{res['level']}] \n File_path : {res['file_path']} \n \n Content : {res['content']}\n")
    
print("\n=== Employment tenure Validation ===")
results = htpvs.hybrid_search(
    "why does the employment tenure is classified as recently_hired?",
)
for res in results:
    print(f"Score: [{res['score']:.3f}] \n level: [{res['level']}] \n File_path : {res['file_path']} \n Content : {res['content']}\n")
"""

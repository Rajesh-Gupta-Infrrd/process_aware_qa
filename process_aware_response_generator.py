
import os
import json
import re
from typing import List, Dict
from openai import OpenAI
from htp_vec_store import htp_vector_store
from transformers import AutoTokenizer

class ProcessAwareResponseGenerator:
    """
    Handles fetching process details and logs related to a mortgage inquiry.
    """

    def __init__(self, api_key: str ,  log_dir: str , milvus_uri :str):
        """
        Initializes the handler with file paths and API credentials.

        Args:
            process_file (str): Path to the process JSON file.
            log_file (str): Path to the log file.
            api_key (str): API key for the OpenAI model.
        """
        self.log_dir = log_dir
        self.llm_client = OpenAI(api_key=api_key)
        self.milvus_uri=milvus_uri
        self.htp_store=htp_vector_store(milvus_uri=self.milvus_uri,htp_dir="./workflows",load_local=True)
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        

    def get_relevent_htps_and_logs(self , user_query : str, top_k: int = 3, step_threshold: float = 0.4, token_threshold: int = 500) -> Dict:
        """
        Loads the mortgage htp data from milvus collection.

        Returns:
            Dict: The loaded htp details.
        """
        #self.htp_store.add_htps("./new_workflows","./workflows")
        
        # Step-level search
        step_results = self.htp_store.hybrid_search(query=user_query, top_k=top_k, depth=2)
        if not step_results:
            print("❌ No step-level results found.")
            return [],[]

        # Filter steps by threshold
        filtered_steps = [step for step in step_results if step["score"] >= step_threshold]
        if not filtered_steps:
            filtered_steps=step_results

        # Sort steps by score in descending order
        filtered_steps.sort(key=lambda x: x["score"], reverse=True)

        # Iteratively remove low scoring steps if they differ significantly from the top step
        final_steps = [filtered_steps[0]]
        for step in filtered_steps[1:]:
            diff = filtered_steps[0]["score"] - step["score"]
            if diff >= 0.08:
                print(f"\n⚠️ Removed step with score {step['score']:.2f} due to large score drop ({diff:.2f})")
                break  # Stop at first large drop
            final_steps.append(step)


        final_results = []
        final_logs=[]
        for step in final_steps:
            step_context = {
                "level": "STEP",
                "score": step["score"],
                "file_path": step["file_path"],
                "lineage": step["lineage"],
                "content": step["content"]
            }

            # Extract lineage info for the step
            lineage = step["lineage"]
            rule_name = lineage.get("rule")
            step_name = lineage.get("step")
            file_path = step["file_path"]
            match = re.search(r"workflow_(\d+)\.json", file_path)
            logs = []
            if match:
                num = match.group(1)
                log_file_name = f"annie-core-backend_{num}.log"
                log_file_path = os.path.join(self.log_dir, log_file_name)
                if os.path.exists(log_file_path):
                    try:
                        with open(log_file_path, "r", encoding="utf-8") as file:
                            logs = file.readlines()
                    except Exception:
                        continue  # Ignore file read errors silently

            # Construct search expression for subtasks
            expr = (
                f'depth == 0 and lineage["rule"] == "{rule_name}" and '
                f'lineage["step"] == "{step_name}"'
            )
            print("Step:\n",step_name)
            # Run subtask-level search using collection API
            subtask_hits = self.htp_store.hybrid_search(query=user_query, top_k=top_k, expr=expr)
            # Sort steps by score in descending order
            subtask_hits.sort(key=lambda x: x["score"], reverse=True)
            selected_subtasks = []
            selected_subtasks_logs=[]
            token_sum = 0
            for sub in subtask_hits:
                task_name = sub["lineage"].get("task")
                if not task_name:
                    continue

                task_logs = []
                inside_block = False

                for line in logs:
                    if f"Executing Task node: {task_name}" in line:
                        inside_block = True
                        continue
                    elif inside_block and "Executing Task node:" in line:
                        break
                    elif inside_block and "## CODE LOG ##" in line:
                        match_log = re.search(r"## CODE LOG ##\s*(.+)", line)
                        if match_log:
                            task_logs.append(match_log.group(1).strip())

                log_text = " ".join(task_logs)
                token_count = len(self.tokenizer.tokenize(log_text))
                if token_count >= token_threshold:
                    print(f"Skipped subtasks due to high token count ({token_count}")
                    break
                token_sum += token_count
                selected_subtasks.append(sub)
                selected_subtasks_logs.append({"Task":task_name,"Logs":task_logs})
                if token_sum >= token_threshold:
                    break

            if selected_subtasks and token_sum <= token_threshold:
                for sub in selected_subtasks:
                    final_results.append({
                        "level": "SUBTASK",
                        "score": sub["score"],
                        "file_path": sub["file_path"],
                        "lineage": sub["lineage"],
                        "content": sub["content"]
                    })
                for sub in selected_subtasks_logs:
                    final_logs.append({
                        "Task":sub["Task"],
                        "Logs":sub["Logs"]

                    })
            else :
                final_results.append(step_context)
                pattern = r"## CODE LOG ##\s*(.+)"
                relevant_logs=[]
                for line in logs:
                    match = re.search(pattern, line)
                    if match:
                        relevant_logs.append(match.group(1).strip())
                final_logs.append({
                    "Step":step_name,
                    "Logs":relevant_logs
                })

        return final_results,final_logs
    
    def create_prompt(self, user_query: str, process_data: Dict, logs: List[str]) -> str:
        """
        Creates an optimal prompt for the LLM.

        Args:
            user_query (str): The user’s question about their mortgage process.
            process_data (Dict): The process details from the JSON file.
            logs (List[str]): The filtered relevant logs.

        Returns:
            str: The constructed prompt.
        """
        prompt=f"""
            You are an expert mortgage process assistant. Your task is to provide detailed explainations to user query using the process details and system logs.
            User Query: '{user_query}'\n
            
            process details:\n{json.dumps(process_data, indent=2)}\n
            
            Relevant system logs:\n{json.dumps(logs, indent=2)}\n

            ANALYSIS INSTRUCTIONS:
            -Identify the key process stages from the workflow data
            -Correlate log entries with each workflow step
            -Explain what happened in natural language 

            RESPONSE GUIDELINESS
            1. **Answer strictly based on the provided process details and logs.**
            2. **Avoid speculating beyond the given logs and process data.**
            3. Don't mention "logs" or "data" - respond naturally.
            4. **If no relevant information is found, simply state: "I do not have relevant information for this query."**
            5. If user query is about process and not related to their Logs then **Don't add any content about their Logs** 
            5. **Keep the response brief, clear, and professional.**
            6. Only explain what is relevant to the user's query.Do not include additional information or reasoning unless explicitly asked in the query. 
        """
        return prompt
        
    
    def generate_response(self, user_query: str) -> str:
        """
        Generates a response based on the user query, process data, and logs.

        Args:
            user_query (str): The mortgage-related user query.

        Returns:
            str: Response generated by the LLM.
        """
        process_data, relevant_logs = self.get_relevent_htps_and_logs(user_query)
        if not process_data:
            return "I do not have relevant information for this query."
        print("\nProcess Data:\n",process_data)
        print("\nLogs:\n",relevant_logs)
        # Construct prompt
        prompt = self.create_prompt(user_query, process_data, relevant_logs)

        # Generate response using LLM
        response = self.llm_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}]
        )

        return response.choices[0].message.content.strip()



# Example usage

if __name__ == "__main__":
    log_dir = "./logs"
    api_key = os.environ["OPENAI_API_KEY"]  # Replace with your API key
    responseGenerator = ProcessAwareResponseGenerator(
        api_key=api_key,
        log_dir=log_dir,
        milvus_uri="htps_vec_store.db"
    )

    user_queries = [
        "Why was my employment tenure classified as recently hired?",
        "Was my Schedule C business process validated correctly?",
        "Does the application and employment years extracted correctly?",
        "How was my business cash flow trend calculated?",
        "Is Schedule C self employment history requirement meet?",
        "Why was I told that tax returns are missing even though I submitted them?",
        "Why my application data is considered as mid year? How did that affect the process?",
        "Can you explain what happens in determine_disbursement_date_scenario?",
        "Did the Schedule C validation step calculate mileage and depreciation correctly?",
        "What logic was applied in the 'htp_schedule_c_validation' for checking my business earnings?"
    ]

    for idx, user_query in enumerate(user_queries, start=1):
        print(f"\nQuery {idx}: {user_query}")
        response = responseGenerator.generate_response(user_query)
        print("Response:", response)


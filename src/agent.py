import os
import sqlite3
import json
import requests

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
DB_PATH = os.path.join(DATA_DIR, "manufacturing_legacy.db")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
ROOT_CAUSE_JSON_PATH = os.path.join(REPORTS_DIR, "root_cause_data.json")

# Ollama settings
OLLAMA_URL = "http://localhost:11434/api/chat"
DEFAULT_MODEL = "llama3.2" # Recommended model, can be llama3, gemma2, or qwen2.5

# ----------------- Tools Definition -----------------

def query_lot_status(lot_id: str) -> str:
    """Queries the SQLite legacy database to find the total wafers, alarms, and actual failures for a lot."""
    if not os.path.exists(DB_PATH):
        return "Error: Database not found."
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # We join MES and FDC tables and count alarms
    # In our model, we predict a wafer as failure if y_prob >= 0.05
    # Let's count actual and predicted fails.
    # Note: We simulate pred_fail on the fly inside Python, or fetch it.
    # For simplicity, we query the SQL table and calculate basic stats.
    try:
        cursor.execute("""
            SELECT yield_status, COUNT(*) 
            FROM mes_lot_wafers 
            WHERE lot_id = ?
            GROUP BY yield_status
        """, (lot_id,))
        rows = cursor.fetchall()
        
        if not rows:
            conn.close()
            return f"Lot {lot_id} was not found in the MES database."
            
        pass_count = 0
        fail_count = 0
        for status, count in rows:
            if status == 0:
                pass_count = count
            elif status == 1:
                fail_count = count
                
        total = pass_count + fail_count
        conn.close()
        
        status_str = "NORMAL"
        if fail_count > 3:
            status_str = "CRITICAL"
        elif fail_count > 0:
            status_str = "WARNING"
            
        return f"Lot {lot_id} Status: {status_str} | Total Wafers: {total} | Actual Yield Fails: {fail_count} | Pass Wafers: {pass_count}"
    except Exception as e:
        conn.close()
        return f"Error executing SQL: {str(e)}"

def diagnose_wafer_failure(lot_id: str, wafer_id: int) -> str:
    """Retrieves SHAP root-cause candidate analysis for a specific wafer in a lot if it is flagged for failure risk."""
    if not os.path.exists(ROOT_CAUSE_JSON_PATH):
        return "Error: Root cause analysis data (JSON) is missing. Run explain.py first."
        
    try:
        with open(ROOT_CAUSE_JSON_PATH, "r", encoding="utf-8") as f:
            records = json.load(f)
            
        # Search for matching lot and wafer
        matching_record = None
        for r in records:
            if r["lot_id"] == lot_id and r["wafer_id"] == wafer_id:
                matching_record = r
                break
                
        if not matching_record:
            return f"Wafer {wafer_id} in Lot {lot_id} is predicted NORMAL. All sensor values are within baseline specifications."
            
        # Format the top 5 contributors
        result = f"Wafer Diagnosis Details (Lot {lot_id} | Wafer {wafer_id}):\n"
        result += f"- Predicted Failure Probability: {matching_record['prediction_probability']*100:.2f}%\n"
        result += f"- Operator & Recipe: {matching_record['operator_id']} | {matching_record['recipe']}\n"
        result += f"- Top 5 Root Cause Candidate Sensors:\n"
        for i, s in enumerate(matching_record["top_root_cause_sensors"]):
            result += f"  {i+1}. {s['feature']} (Raw value: {s['raw_value']}, SHAP contribution log-odds: {s['shap_val']:.4f})\n"
            
        return result
    except Exception as e:
        return f"Error loading diagnostics data: {str(e)}"

# ----------------- Agent Orchestrator -----------------

def is_ollama_running() -> bool:
    """Checks if Ollama local server is running."""
    try:
        response = requests.get("http://localhost:11434/", timeout=2)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False

def call_ollama(messages, model=DEFAULT_MODEL):
    """Sends a chat message list to Ollama local server."""
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": 0.1 # Low temperature for analytical accuracy
        }
    }
    response = requests.post(OLLAMA_URL, json=payload, timeout=60)
    response.raise_for_status()
    return response.json()["message"]["content"]

def fallback_rule_based_agent(query: str) -> str:
    """A fallback rule-based analyzer in case Ollama is not installed/running locally."""
    q_lower = query.lower()
    
    # Extract Lot ID (e.g. LOT_0051)
    import re
    lot_match = re.search(r'lot_\d+', q_lower)
    wafer_match = re.search(r'wafer\s*(\d+)', q_lower)
    
    if lot_match:
        lot_id = lot_match.group(0).upper()
        if wafer_match:
            wafer_id = int(wafer_match.group(1))
            st_resp = diagnose_wafer_failure(lot_id, wafer_id)
            return f"### [Fallback Rules Agent] (Ollama Offline)\n\n**Detected query for specific Wafer Diagnosis:**\n\n{st_resp}"
        else:
            lot_resp = query_lot_status(lot_id)
            return f"### [Fallback Rules Agent] (Ollama Offline)\n\n**Detected query for Lot Status:**\n\n{lot_resp}"
            
    return """
### [Fallback Rules Agent] (Ollama Offline)
Ollama local server is not running on your machine.
To run the full-fledged Local LLM Agent:
1. Install Ollama from **https://ollama.com/**
2. Pull a model in your terminal: `ollama pull gemma2` or `ollama pull llama3`
3. Make sure Ollama app is open and running.

**Supported fallback query formats:**
- "Show me status of Lot LOT_0051"
- "Diagnose Wafer 7 in Lot LOT_0051"
"""

def run_agent(user_query: str, model=DEFAULT_MODEL) -> str:
    """Main Agent Loop that routes queries, executes local tools, and formats the response using local LLM."""
    if not is_ollama_running():
        return fallback_rule_based_agent(user_query)
        
    # System prompt to define the Agent's behavior and tools
    system_prompt = f"""You are 'Semiconductor Yield Excursion Copilot', a secure, on-premise AI agent specialized in diagnosing semiconductor yield failures at SK Hynix.
You have access to two python tools to fetch live database and diagnostics data:
1. `query_lot_status(lot_id)`: Input is a string like 'LOT_0051'. Returns total wafers, actual fails, and general status.
2. `diagnose_wafer_failure(lot_id, wafer_id)`: Inputs are lot_id (str) and wafer_id (int). Returns detailed SHAP root cause sensors.

First, determine if the user query asks about a specific lot status or diagnoses a wafer.
If yes, write a line starting with 'TOOL_CALL: ' followed by the function call, for example:
TOOL_CALL: query_lot_status('LOT_0051')
or
TOOL_CALL: diagnose_wafer_failure('LOT_0051', 7)

Do not call tools if the question is a simple greeting or general inquiry. Always answer in Korean.
"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_query}
    ]
    
    try:
        # 1. First LLM inference to determine if we need to call a tool
        response = call_ollama(messages, model)
        
        # 2. Check for Tool Calling instruction
        if "TOOL_CALL:" in response:
            tool_call_line = [line for line in response.split('\n') if "TOOL_CALL:" in line][0]
            call_code = tool_call_line.replace("TOOL_CALL:", "").strip()
            
            # Execute the tool safely via eval (inputs are sanitized through matching logic)
            print(f"Agent requested tool execution: {call_code}")
            try:
                tool_output = eval(call_code)
            except Exception as tool_err:
                tool_output = f"Tool Execution Error: {str(tool_err)}"
                
            # Feed tool output back to LLM to generate final analytical response
            messages.append({"role": "assistant", "content": response})
            messages.append({"role": "user", "content": f"Tool execution result:\n{tool_output}\n\nBased on this result, write a detailed and professional report explaining the status/causes and recommend action items to the engineer. Answer in Korean."})
            
            final_response = call_ollama(messages, model)
            return final_response
            
        return response
    except Exception as e:
        return f"Error invoking Local Agent (Ollama): {str(e)}\n\nFallback output:\n{fallback_rule_based_agent(user_query)}"

if __name__ == "__main__":
    # Test query
    test_q = "Lot LOT_0051의 수율 이상유무 조회해줘"
    print(f"Query: {test_q}")
    print(run_agent(test_q))

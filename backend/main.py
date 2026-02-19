from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
import json, io, re, uuid, os
from google import genai
import plotly.express as px
import plotly.graph_objects as go
import plotly.utils

client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

app = FastAPI(title="DataWhisper API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

sessions = {}

class QueryRequest(BaseModel):
    session_id: str
    question: str

class QueryResponse(BaseModel):
    answer: str
    code: str
    chart: dict | None = None
    table: dict | None = None
    error: str | None = None

def df_summary(df):
    shape = f"Shape: {df.shape[0]} rows x {df.shape[1]} columns"
    columns = f"Columns: {list(df.columns)}"
    dtypes = f"Dtypes:\n{df.dtypes.to_string()}"
    sample = f"Sample:\n{df.head(5).to_string()}"
    nulls = f"Nulls:\n{df.isnull().sum().to_string()}"
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    stats = f"Stats:\n{df[numeric_cols].describe().to_string()}" if numeric_cols else ""
    return f"{shape}\n{columns}\n{dtypes}\n{sample}\n{nulls}\n{stats}"

def extract_code(text):
    m = re.search(r"```python\n(.*?)```", text, re.DOTALL)
    if m: return m.group(1).strip()
    m = re.search(r"```\n(.*?)```", text, re.DOTALL)
    if m: return m.group(1).strip()
    return text.strip()

def safe_execute(code, df):
    local_vars = {"df": df.copy(), "pd": pd, "px": px, "go": go, "result": None, "fig": None}
    try:
        exec(code, {"__builtins__": __builtins__}, local_vars)
    except Exception as e:
        return None, None, None, str(e)
    result = local_vars.get("result")
    fig = local_vars.get("fig")
    chart_json = None
    if fig is not None:
        try:
            chart_json = json.loads(plotly.utils.PlotlyJSONEncoder().encode(fig))
        except:
            pass
    table_data = None
    if isinstance(result, pd.DataFrame):
        table_data = {"columns": list(result.columns), "rows": result.head(100).values.tolist()}
        result = None
    elif isinstance(result, pd.Series):
        table_data = {"columns": [result.name or "value"], "rows": [[v] for v in result.head(100).values.tolist()]}
        result = None
    elif isinstance(result, (list, tuple)) and len(result) >= 2:
        # Convert list with 2+ items to a table
        table_data = {"columns": ["value"], "rows": [[str(v)] for v in result[:100]]}
        result = None
    elif isinstance(result, dict) and len(result) >= 2:
        # Convert dict with 2+ keys to a table
        table_data = {"columns": ["column", "value"], "rows": [[str(k), str(v)] for k, v in list(result.items())[:100]]}
        result = None
    return result, chart_json, table_data, None

def build_final_answer(explanation, result, question):
    """
    Build a clean final answer.
    If result is a simple number, IGNORE the AI explanation entirely
    and build a simple clean sentence using the real computed value.
    """
    if result is None:
        return explanation

    if isinstance(result, (int, float, np.integer, np.floating)):
        result = float(result) if isinstance(result, (np.floating, float)) else int(result)
        formatted = f"{result:,}" if isinstance(result, int) else f"{result:,.2f}"
        q = question.lower()
        # Build a natural answer based on the question type
        if any(w in q for w in ["how many rows", "row count", "number of rows"]):
            return f"Your dataset has {formatted} rows."
        elif any(w in q for w in ["how many columns", "column count", "number of columns"]):
            return f"Your dataset has {formatted} columns."
        elif any(w in q for w in ["unique", "distinct"]):
            return f"There are {formatted} unique values."
        elif any(w in q for w in ["average", "mean"]):
            return f"The average is {formatted}."
        elif any(w in q for w in ["max", "maximum", "highest", "largest"]):
            return f"The maximum value is {formatted}."
        elif any(w in q for w in ["min", "minimum", "lowest", "smallest"]):
            return f"The minimum value is {formatted}."
        elif any(w in q for w in ["sum", "total"]):
            return f"The total is {formatted}."
        else:
            return f"The answer is {formatted}."

    # For dicts — handle specially
    if isinstance(result, dict):
        if 'column_names' in result:
            cols = result['column_names']
            dtypes = result.get('column_dtypes', {})
            lines = ", ".join([f"{c} ({str(dtypes.get(c, 'unknown')).replace('dtype(', '').replace(')', '').replace(chr(39), '')})" for c in cols])
            return f"Your dataset has {len(cols)} columns: {lines}"
        return explanation

    # For lists/arrays, show them cleanly
    if hasattr(result, "__iter__") and not isinstance(result, str):
        items = list(result)
        clean = ", ".join(str(i) for i in items)
        return f"The values are: {clean}"

    # For strings
    return f"{explanation} {result}"

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided.")
    ext = file.filename.rsplit(".", 1)[-1].lower()
    if ext not in ("csv", "xlsx", "xls"):
        raise HTTPException(status_code=400, detail="Only CSV/Excel supported.")
    contents = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(contents)) if ext == "csv" else pd.read_excel(io.BytesIO(contents))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not parse file: {e}")
    session_id = str(uuid.uuid4())
    sessions[session_id] = {"df": df, "filename": file.filename, "summary": df_summary(df), "history": []}
    return {"session_id": session_id, "filename": file.filename, "rows": df.shape[0], "columns": df.shape[1], "column_names": list(df.columns), "preview": df.head(5).to_dict(orient="records")}

@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    session = sessions.get(request.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found.")
    df = session["df"]
    summary = session["summary"]
    history = session["history"]

    system_prompt = f"""You are DataWhisper, an expert data analyst AI.
Dataset summary:
{summary}

Rules:
- df is already loaded. Never reload it.
- Use only pd, px, go. No matplotlib. No other imports.
- Always assign the final computed answer to a variable called result.
- For charts: assign to fig using px or go with template=plotly_dark and color_discrete_sequence=["#00e5ff"].
- When creating bar charts, always use value_counts() for categorical columns.
- Always wrap code in ```python``` blocks.
- After the code block, write ONE short friendly sentence explaining what you calculated.
- Do NOT include any numbers in your explanation. Just describe what was calculated, e.g. "I counted the unique values in the First Name column."
- NEVER show Python objects like dtype(), Index() in your answer."""

    conversation = "\n".join([f"{h['role']}: {h['content']}" for h in history[-4:]])
    full_prompt = f"{system_prompt}\n\nPrevious conversation:\n{conversation}\n\nuser: {request.question}"

    try:
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=full_prompt
        )
        llm_response = response.text
    except Exception as e:
        print(f"REAL ERROR: {e}")
        raise HTTPException(status_code=500, detail=f"LLM error: {e}")

    code = extract_code(llm_response)
    explanation = re.sub(r"```.*?```", "", llm_response, flags=re.DOTALL).strip() or "Here is the result."
    result, chart_json, table_data, exec_error = safe_execute(code, df)

    # Save only clean explanation in history — no numbers to confuse future answers
    session["history"].append({"role": "user", "content": request.question})
    session["history"].append({"role": "assistant", "content": explanation})

    if exec_error:
        return QueryResponse(answer=explanation, code=code, error=f"Execution error: {exec_error}")

    # Build answer using ACTUAL computed result, not AI's guess
    answer = build_final_answer(explanation, result, request.question)

    return QueryResponse(answer=answer, code=code, chart=chart_json, table=table_data)

@app.get("/health")
async def health():
    return {"status": "ok"}
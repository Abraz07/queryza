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

app = FastAPI(title="Queryza API")
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
    return ""

def safe_execute(code, df):
    if not code.strip():
        return None, None, None, None
    local_vars = {"df": df.copy(), "pd": pd, "np": np, "px": px, "go": go, "result": None, "fig": None}
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
        table_data = {"columns": list(result.columns), "rows": [[str(v) for v in row] for row in result.head(100).values.tolist()]}
        result = None
    elif isinstance(result, pd.Series):
        table_data = {"columns": [str(result.name or "value"), "count"], "rows": [[str(k), str(v)] for k, v in result.head(100).items()]}
        result = None
    elif isinstance(result, (list, tuple)) and len(result) >= 2:
        table_data = {"columns": ["value"], "rows": [[str(v)] for v in result[:100]]}
        result = None
    elif isinstance(result, dict) and len(result) >= 2:
        # Special case: column names + dtypes dict
        if "column_names" in result and "column_dtypes" in result:
            cols = result["column_names"]
            dtypes = result["column_dtypes"]
            table_data = {
                "columns": ["Column Name", "Data Type"],
                "rows": [[str(c), str(dtypes.get(c, "unknown")).replace("dtype(", "").replace(")", "").replace("'", "").replace("object", "text").replace("int64", "number").replace("float64", "decimal").replace("StringDtype(storage=python, na_value=nan)", "text")] for c in cols]
            }
        else:
            table_data = {"columns": ["column", "value"], "rows": [[str(k), str(v)] for k, v in list(result.items())[:100]]}
        result = None
    return result, chart_json, table_data, None

def is_conversational(question):
    """Detect questions that don't need code — answer directly from metadata."""
    q = question.lower().strip()
    patterns = [
        "what is this", "what's this", "whats this", "what is the file",
        "tell me about", "describe this", "what does this", "overview",
        "summarize", "what kind", "what type", "about this file",
        "about this dataset", "about this data", "what are we looking at",
    ]
    return any(p in q for p in patterns)

def is_simple_metadata(question, df):
    """Answer simple row/column count questions directly without AI."""
    q = question.lower().strip()
    if any(w in q for w in ["how many rows", "number of rows", "row count", "how many records"]):
        return f"Your dataset has {df.shape[0]:,} rows."
    if any(w in q for w in ["how many columns", "number of columns", "column count", "how many fields"]):
        return f"Your dataset has {df.shape[1]} columns."
    if any(w in q for w in ["column names", "columns and", "what are the columns", "data types", "datatypes"]):
        type_map = {"int64": "Number", "float64": "Decimal", "object": "Text", "bool": "Boolean", "datetime64[ns]": "Date"}
        rows = [[col, type_map.get(str(df[col].dtype), "Text")] for col in df.columns]
        return {"__table__": True, "columns": ["Column Name", "Data Type"], "rows": rows, "message": f"Your dataset has {df.shape[1]} columns:"}
    return None

def build_final_answer(explanation, result, question):
    if result is None:
        return explanation
    if isinstance(result, (int, float, np.integer, np.floating)):
        result = float(result) if isinstance(result, (np.floating, float)) else int(result)
        formatted = f"{result:,}" if isinstance(result, int) else f"{result:,.2f}"
        q = question.lower()
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
    if hasattr(result, "__iter__") and not isinstance(result, str):
        items = list(result)
        clean = ", ".join(str(i) for i in items)
        return f"The values are: {clean}"
    return f"{explanation}\n\n{result}" if result else explanation

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

    # Answer simple metadata questions directly — no AI needed
    simple = is_simple_metadata(request.question, df)
    if simple:
        if isinstance(simple, dict) and simple.get("__table__"):
            table_data = {"columns": simple["columns"], "rows": simple["rows"]}
            answer = simple["message"]
            session["history"].append({"role": "user", "content": request.question})
            session["history"].append({"role": "assistant", "content": answer})
            return QueryResponse(answer=answer, code="", table=table_data)
        session["history"].append({"role": "user", "content": request.question})
        session["history"].append({"role": "assistant", "content": simple})
        return QueryResponse(answer=simple, code="")

    # For conversational questions, skip code and answer directly
    if is_conversational(request.question):
        conversational_prompt = f"""The user uploaded a file called '{session["filename"]}' with this summary:
{summary}

The user asked: "{request.question}"

Write a SHORT, FRIENDLY, plain English description of what this dataset is about. 
- Mention the number of rows and columns
- Mention what kind of data it contains based on the column names
- Keep it to 3-4 sentences max
- Do NOT use technical jargon or Python terms
- Do NOT write any code"""
        try:
            response = client.models.generate_content(model="gemini-2.0-flash", contents=conversational_prompt)
            answer = response.text.strip()
        except Exception as e:
            answer = f"This dataset has {df.shape[0]:,} rows and {df.shape[1]} columns with the following fields: {', '.join(df.columns.tolist())}."
        session["history"].append({"role": "user", "content": request.question})
        session["history"].append({"role": "assistant", "content": answer})
        return QueryResponse(answer=answer, code="")

    # For analytical questions, generate and run code
    system_prompt = f"""You are DataWhisper, an expert data analyst AI.
Dataset: '{session["filename"]}'
Summary:
{summary}

Rules:
- df is already loaded. NEVER reload it.
- Use only pd, np, px, go. No matplotlib. No other imports.
- ALWAYS assign the final answer to result. NEVER use print(). NEVER leave result as None.
- For charts: assign to fig using px or go with template=plotly_dark and color_discrete_sequence=["#6366f1"].
- When making bar charts, use value_counts() for categorical columns.
- Always wrap code in ```python``` blocks.
- After the code, write ONE sentence describing what you calculated. Do NOT include numbers — just describe the calculation.
- Example good explanation: "I counted the number of unique values in the First Name column."
- Example bad explanation: "There are 690 unique values." (never put numbers in explanation)"""

    conversation = "\n".join([f"{h['role']}: {h['content']}" for h in history[-4:]])
    full_prompt = f"{system_prompt}\n\nPrevious conversation:\n{conversation}\n\nuser: {request.question}"

    try:
        response = client.models.generate_content(model="gemini-2.0-flash", contents=full_prompt)
        llm_response = response.text
    except Exception as e:
        print(f"REAL ERROR: {e}")
        raise HTTPException(status_code=500, detail=f"LLM error: {e}")

    code = extract_code(llm_response)
    explanation = re.sub(r"```.*?```", "", llm_response, flags=re.DOTALL).strip() or "Here is the result."
    result, chart_json, table_data, exec_error = safe_execute(code, df)

    session["history"].append({"role": "user", "content": request.question})
    session["history"].append({"role": "assistant", "content": explanation})

    if exec_error:
        return QueryResponse(answer=explanation, code=code, error=f"Execution error: {exec_error}")

    if result is None and table_data is not None:
        answer = "Here's what I found:"
    elif result is None and chart_json is not None:
        answer = "Here's the chart:"
    else:
        answer = build_final_answer(explanation, result, request.question)

    return QueryResponse(answer=answer, code=code, chart=chart_json, table=table_data)

@app.get("/health")
async def health():
    return {"status": "ok"}
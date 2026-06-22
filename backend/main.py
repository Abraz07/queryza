from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import pandas as pd
import numpy as np
import json, io, re, uuid, os
from google import genai
import plotly.express as px
import plotly.graph_objects as go
import plotly.utils

client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

MAX_FILE_BYTES = 25 * 1024 * 1024
MAX_ROWS = 100_000
MAX_COLUMNS = 500
MAX_CODE_RETRIES = 2
LLM_MODELS = ["gemini-2.0-flash", "gemini-1.5-flash", "gemini-2.0-flash-lite"]

app = FastAPI(title="Queryza API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    if isinstance(exc, HTTPException):
        return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})
    print(f"Unhandled error on {request.url.path}: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error. Please try a smaller file or a different sheet."},
    )

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

def generate_suggestions(df):
    suggestions = [
        "What is this dataset about?",
        "What are the column names and their data types?",
        f"How many rows are in this dataset?",
    ]
    numeric = df.select_dtypes(include="number").columns.tolist()
    categorical = df.select_dtypes(include=["object", "category", "string"]).columns.tolist()
    datetime_cols = df.select_dtypes(include=["datetime64", "datetimetz"]).columns.tolist()

    null_cols = [c for c in df.columns if df[c].isnull().any()]
    if null_cols:
        suggestions.append(f"Which columns have missing values?")

    if numeric:
        suggestions.append(f"What is the average of '{str(numeric[0])}'?")
        suggestions.append("Show me a summary of all numeric columns")

    for col in categorical:
        nunique = df[col].nunique(dropna=True)
        if 2 <= nunique <= 40:
            suggestions.append(f"Show me a bar chart of '{str(col)}'")
            break

    if datetime_cols:
        suggestions.append(f"How does the data trend over '{str(datetime_cols[0])}'?")
    elif len(categorical) >= 2:
        suggestions.append(f"Break down '{str(categorical[0])}' by '{str(categorical[1])}'")

    seen = set()
    unique = []
    for s in suggestions:
        if s not in seen:
            seen.add(s)
            unique.append(s)
    return unique[:6]

def validate_dataframe(df):
    if df.shape[0] > MAX_ROWS:
        raise HTTPException(
            status_code=400,
            detail=f"Dataset has {df.shape[0]:,} rows — maximum is {MAX_ROWS:,}. Try a smaller file or sample.",
        )
    if df.shape[1] > MAX_COLUMNS:
        raise HTTPException(
            status_code=400,
            detail=f"Dataset has {df.shape[1]} columns — maximum is {MAX_COLUMNS}.",
        )

def load_dataframe(contents: bytes, ext: str, sheet: str | None = None) -> pd.DataFrame:
    buf = io.BytesIO(contents)
    if ext == "csv":
        return pd.read_csv(buf)
    return pd.read_excel(buf, sheet_name=sheet or 0)

def preview_records(df: pd.DataFrame) -> list[dict]:
    preview = df.head(5).copy()
    for col in preview.columns:
        if pd.api.types.is_datetime64_any_dtype(preview[col]):
            preview[col] = preview[col].astype(str)
    records = preview.astype(object).where(pd.notnull(preview), None).to_dict(orient="records")
    safe = []
    for row in records:
        safe.append({
            k: (None if v is None else v.item() if hasattr(v, "item") else str(v) if not isinstance(v, (str, int, float, bool)) else v)
            for k, v in row.items()
        })
    return safe

def create_session(filename: str, df: pd.DataFrame, sheet: str | None = None):
    session_id = str(uuid.uuid4())
    display_name = f"{filename} ({sheet})" if sheet else filename
    sessions[session_id] = {
        "df": df,
        "filename": display_name,
        "summary": df_summary(df),
        "history": [],
    }
    return {
        "session_id": session_id,
        "filename": display_name,
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "column_names": [str(c) for c in df.columns],
        "preview": preview_records(df),
        "suggestions": generate_suggestions(df),
        "sheet": sheet,
    }

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
    q = question.lower().strip()
    patterns = [
        "what is this", "what's this", "whats this", "what is the file",
        "tell me about", "describe this", "what does this", "overview",
        "summarize", "what kind", "what type", "about this file",
        "about this dataset", "about this data", "what are we looking at",
    ]
    return any(p in q for p in patterns)

def is_simple_metadata(question, df):
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

def find_mentioned_column(question: str, df: pd.DataFrame):
    for col in df.columns:
        col_str = str(col)
        if f"'{col_str}'" in question or f'"{col_str}"' in question or col_str in question:
            return col
        if col_str.lower() in question.lower():
            return col
    return None

def numeric_summary_table(df: pd.DataFrame) -> dict:
    numeric = df.select_dtypes(include="number")
    desc = numeric.describe()
    columns = ["Stat"] + [str(c) for c in desc.columns]
    rows = []
    for stat in desc.index:
        rows.append([str(stat)] + [
            f"{v:,.4f}" if pd.notna(v) and isinstance(v, (float, np.floating)) else str(v)
            for v in desc.loc[stat]
        ])
    return {"columns": columns, "rows": rows}

def try_analytical_fast_path(question: str, df: pd.DataFrame):
    q = question.lower()
    numeric = df.select_dtypes(include="number")

    if ("summary" in q or "describe" in q) and ("numeric" in q or "number" in q or "decimal" in q):
        if numeric.empty:
            return "No numeric columns found in this dataset."
        return {
            "__table__": True,
            **numeric_summary_table(df),
            "message": "Summary of numeric columns:",
        }

    col = find_mentioned_column(question, df)
    if col is not None and col in numeric.columns:
        series = numeric[col]
        if any(w in q for w in ["average", "mean"]):
            return f"The average of '{col}' is {series.mean():,.4f}."
        if any(w in q for w in ["sum", "total"]):
            return f"The total of '{col}' is {series.sum():,.4f}."
        if any(w in q for w in ["max", "maximum", "highest", "largest"]):
            return f"The maximum of '{col}' is {series.max():,.4f}."
        if any(w in q for w in ["min", "minimum", "lowest", "smallest"]):
            return f"The minimum of '{col}' is {series.min():,.4f}."
        if any(w in q for w in ["unique", "distinct", "count"]):
            return f"'{col}' has {series.nunique():,} unique values."

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

def extract_llm_text(response) -> str:
    try:
        text = response.text
        if text and text.strip():
            return text.strip()
    except (ValueError, AttributeError, TypeError):
        pass
    for candidate in getattr(response, "candidates", None) or []:
        content = getattr(candidate, "content", None)
        if not content:
            continue
        parts = getattr(content, "parts", None) or []
        texts = [p.text for p in parts if getattr(p, "text", None)]
        if texts:
            return "\n".join(texts).strip()
    raise ValueError("Model returned no text. Please try again.")

def generate_llm_code(system_prompt: str, user_prompt: str) -> str:
    prompt = f"{system_prompt}\n\n{user_prompt}"
    last_err = None
    for model in LLM_MODELS:
        try:
            response = client.models.generate_content(model=model, contents=prompt)
            return extract_llm_text(response)
        except Exception as e:
            last_err = e
            print(f"LLM {model} failed: {e}")
    raise HTTPException(status_code=500, detail=f"AI service unavailable: {last_err}")

def run_analytical_query(session, question: str) -> QueryResponse:
    df = session["df"]
    summary = session["summary"]
    history = session["history"]

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
    user_prompt = f"Previous conversation:\n{conversation}\n\nuser: {question}"

    try:
        llm_response = generate_llm_code(system_prompt, user_prompt)
    except Exception as e:
        print(f"REAL ERROR: {e}")
        raise HTTPException(status_code=500, detail=f"LLM error: {e}")

    code = extract_code(llm_response)
    explanation = re.sub(r"```.*?```", "", llm_response, flags=re.DOTALL).strip() or "Here is the result."
    result, chart_json, table_data, exec_error = safe_execute(code, df)

    retries = 0
    while exec_error and retries < MAX_CODE_RETRIES:
        retry_prompt = f"""The user asked: "{question}"

You generated this code:
```python
{code}
```

It failed with this error:
{exec_error}

Fix the code. df is already loaded — do not reload it.
Use only pd, np, px, go. Assign the final answer to result.
Always wrap code in ```python``` blocks.
After the code, write ONE sentence describing what you calculated (no numbers in the explanation)."""
        try:
            llm_response = generate_llm_code(system_prompt, retry_prompt)
        except Exception as e:
            break
        code = extract_code(llm_response)
        explanation = re.sub(r"```.*?```", "", llm_response, flags=re.DOTALL).strip() or explanation
        result, chart_json, table_data, exec_error = safe_execute(code, df)
        retries += 1

    session["history"].append({"role": "user", "content": question})
    session["history"].append({"role": "assistant", "content": explanation})

    if exec_error:
        return QueryResponse(answer=explanation, code=code, error=f"Execution error: {exec_error}")

    if result is None and table_data is not None:
        answer = "Here's what I found:"
    elif result is None and chart_json is not None:
        answer = "Here's the chart:"
    else:
        answer = build_final_answer(explanation, result, question)

    return QueryResponse(answer=answer, code=code, chart=chart_json, table=table_data)

@app.post("/upload")
async def upload_file(file: UploadFile = File(...), sheet: str | None = Form(None)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided.")
    ext = file.filename.rsplit(".", 1)[-1].lower()
    if ext not in ("csv", "xlsx", "xls"):
        raise HTTPException(status_code=400, detail="Only CSV/Excel supported.")
    try:
        contents = await file.read()
        if len(contents) > MAX_FILE_BYTES:
            raise HTTPException(
                status_code=400,
                detail=f"File too large ({len(contents) // (1024 * 1024)} MB). Maximum is {MAX_FILE_BYTES // (1024 * 1024)} MB.",
            )
        if ext in ("xlsx", "xls") and not sheet:
            try:
                excel = pd.ExcelFile(io.BytesIO(contents))
                sheets = excel.sheet_names
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Could not read Excel file: {e}")
            if len(sheets) > 1:
                return {"needs_sheet": True, "sheets": sheets, "filename": file.filename}
        df = load_dataframe(contents, ext, sheet)
        validate_dataframe(df)
        return create_session(file.filename, df, sheet)
    except HTTPException:
        raise
    except Exception as e:
        print(f"Upload error: {e}")
        raise HTTPException(status_code=400, detail=f"Could not process file: {e}")

@app.delete("/session/{session_id}")
async def clear_session(session_id: str):
    session = sessions.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found.")
    session["history"] = []
    return {"status": "ok"}

@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    session = sessions.get(request.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found.")
    df = session["df"]
    summary = session["summary"]
    history = session["history"]

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
            response = client.models.generate_content(model=LLM_MODELS[0], contents=conversational_prompt)
            answer = extract_llm_text(response)
        except Exception as e:
            answer = f"This dataset has {df.shape[0]:,} rows and {df.shape[1]} columns with the following fields: {', '.join(str(c) for c in df.columns.tolist())}."
        session["history"].append({"role": "user", "content": request.question})
        session["history"].append({"role": "assistant", "content": answer})
        return QueryResponse(answer=answer, code="")

    fast = try_analytical_fast_path(request.question, df)
    if fast:
        if isinstance(fast, dict) and fast.get("__table__"):
            table_data = {"columns": fast["columns"], "rows": fast["rows"]}
            answer = fast["message"]
            session["history"].append({"role": "user", "content": request.question})
            session["history"].append({"role": "assistant", "content": answer})
            return QueryResponse(answer=answer, code="", table=table_data)
        session["history"].append({"role": "user", "content": request.question})
        session["history"].append({"role": "assistant", "content": fast})
        return QueryResponse(answer=fast, code="")

    try:
        return run_analytical_query(session, request.question)
    except HTTPException:
        raise
    except Exception as e:
        print(f"Query error: {e}")
        raise HTTPException(status_code=500, detail=f"Query failed: {e}")

@app.get("/health")
async def health():
    return {"status": "ok"}

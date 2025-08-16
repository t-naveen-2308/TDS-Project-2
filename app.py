import os
import io
import tarfile
import json
import tempfile
import asyncio
from typing import List

import docker
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse, PlainTextResponse
from starlette.middleware.cors import CORSMiddleware
from groq import Groq

from dotenv import load_dotenv

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL_DEFAULT = os.getenv("GROQ_MODEL_DEFAULT")

if not GROQ_API_KEY:
    print("WARNING: GROQ_API_KEY is not set. Set it in the environment before starting.")

_groq_client = Groq(api_key=GROQ_API_KEY)

def groq_chat(messages, model: str = GROQ_MODEL_DEFAULT, temperature: float = 0.2, max_tokens: int = 1500) -> str:
    resp = _groq_client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return resp.choices[0].message.content

_docker_client = None

def _get_docker_client():
    global _docker_client
    if _docker_client is None:
        _docker_client = docker.from_env()
        try:
            _docker_client.images.get("python:3.11")
        except docker.errors.ImageNotFound:
            _docker_client.images.pull("python:3.11")
    return _docker_client

def call_python_script(script: str, timeout_sec: int = 90) -> str:
    client = _get_docker_client()

    script_name = "script.py"
    tarstream = io.BytesIO()
    with tarfile.open(fileobj=tarstream, mode="w") as tar:
        script_bytes = script.encode("utf-8")
        tarinfo = tarfile.TarInfo(name=script_name)
        tarinfo.size = len(script_bytes)
        tar.addfile(tarinfo, io.BytesIO(script_bytes))
    tarstream.seek(0)

    container = client.containers.create(
        "python:3.11",
        command="sleep infinity",
        detach=True,
    )
    try:
        container.start()
        container.put_archive(path="/tmp", data=tarstream.read())
        exec_id = client.api.exec_create(container.id, cmd=f"python /tmp/{script_name}")
        output = client.api.exec_start(exec_id, detach=False, stream=True)
        collected = []
        loop = 0
        for chunk in output:
            collected.append(chunk)
            loop += 1
        out_bytes = b"".join(collected)
        return out_bytes.decode("utf-8", errors="replace")
    finally:
        try:
            container.remove(force=True)
        except Exception:
            pass

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

README_HINT = "POST multipart: questions.txt (required) + optional files; returns the answer in the requested format."

@app.get("/")
def root():
    return PlainTextResponse(README_HINT)

@app.post("/api/")
async def api(files: List[UploadFile] = File(...)):
    timeout_sec = int(os.getenv("AGENT_TIMEOUT_SEC", "170"))  # leave headroom below 180s
    try:
        return await asyncio.wait_for(handle_request(files), timeout=timeout_sec)
    except asyncio.TimeoutError:
        return JSONResponse({"error": "Timed out"}, status_code=504)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

async def handle_request(files: List[UploadFile]):
    qfile = next((f for f in files if f.filename and f.filename.lower().endswith("questions.txt")), None)
    if not qfile:
        return JSONResponse({"error": "questions.txt is required"}, status_code=400)
    attachments = [f for f in files if f is not qfile]

    with tempfile.TemporaryDirectory() as workdir:
        qpath = os.path.join(workdir, "questions.txt")
        with open(qpath, "wb") as w:
            w.write(await qfile.read())

        saved_files = []
        for f in attachments:
            dst = os.path.join(workdir, f.filename)
            with open(dst, "wb") as w:
                w.write(await f.read())
            saved_files.append(dst)

        with open(qpath, "r", encoding="utf-8", errors="ignore") as r:
            question_text = r.read()

        system = (
            "You are a data analyst agent. Read the question and local files, then respond with ONLY a JSON object:\n"
            '{"steps":[...],'
            '"python_blocks":[{"filename":"run1.py","code":"..."}],'
            '"final_format":"array|object",'
            '"postprocess_instructions":"..."}\n'
            "Rules:\n"
            "- Python blocks must be self-contained: read local files using their basenames relative to the working dir provided.\n"
            "- Perform scraping if required (use requests + BeautifulSoup with a User-Agent).\n"
            "- Compute stats/plots in Python and print a SINGLE JSON object with interim results to stdout.\n"
            "- If a plot is needed, ensure base64 data URI and keep image under 100,000 bytes.\n"
            "- Do not include placeholders; include full runnable code."
        )
        user = (
            f"Question (verbatim):\n{question_text}\n\n"
            f"Working directory: {workdir}\n"
            f"Files: " + ", ".join(os.path.basename(p) for p in saved_files) + "\n"
            "Return strictly the planning JSON. No prose."
        )

        planning_json = groq_chat(
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.1,
            max_tokens=1200,
        )

        try:
            plan = json.loads(planning_json)
        except Exception:
            plan = {"steps": [], "python_blocks": [{"filename": "run.py", "code": planning_json}], "final_format": "array"}

        interim = {}

        for i, block in enumerate(plan.get("python_blocks", [])):
            code = block.get("code") or ""
            safe_cd = f"import os\nos.chdir({json.dumps(workdir)})\n"
            full_code = safe_cd + code
            out = call_python_script(full_code)
            try:
                data = json.loads(out.strip())
                if isinstance(data, dict):
                    interim.update(data)
                else:
                    interim[f"block_{i}_raw"] = out
            except Exception:
                interim[f"block_{i}_raw"] = out

        assembler_system = (
            "You will be given the original question and the interim JSON results from Python runs. "
            "Return ONLY the final answer in the exact schema requested by the question. "
            "If the question asks for a JSON array, return an array; if it asks for a JSON object with specific keys, "
            "return an object with exactly those keys. No prose."
        )
        assembler_user = json.dumps({
            "question": question_text,
            "interim": interim,
            "expected_format": plan.get("final_format", "array"),
            "postprocess_instructions": plan.get("postprocess_instructions", "")
        })

        final_text = groq_chat(
            messages=[
                {"role": "system", "content": assembler_system},
                {"role": "user", "content": assembler_user},
            ],
            temperature=0.0,
            max_tokens=1000,
        )

        try:
            final_obj = json.loads(final_text)
            return JSONResponse(final_obj)
        except Exception:
            return PlainTextResponse(final_text)

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)
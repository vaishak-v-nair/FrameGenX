from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional, Dict, Any
import uvicorn
from pathlib import Path
import json
import base64

from src.pipeline import FrameGenXPipeline
from src.intent.models import ContractV2

app = FastAPI(title="FrameGenX API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For dev; lock down in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

runs_dir = Path("runs")
runs_dir.mkdir(parents=True, exist_ok=True)
app.mount("/runs", StaticFiles(directory="runs"), name="runs")

pipeline = FrameGenXPipeline()

class GenerateRequest(BaseModel):
    prompt: Optional[str] = None
    contract: Optional[ContractV2] = None

@app.get("/")
def read_root():
    return {
        "status": "online",
        "service": "FrameGenX API",
        "version": "0.1.0",
        "docs_url": "/docs"
    }

class GenerateResponse(BaseModel):
    run_id: str
    success: bool
    verification_report: Dict[str, Any]
    image_base64: Optional[str] = None

@app.post("/generate", response_model=GenerateResponse)
def generate_image(request: GenerateRequest):
    if not request.prompt and not request.contract:
        raise HTTPException(status_code=400, detail="Must provide either prompt or contract.")
    
    # In a fully realized production environment, you could skip extraction if contract is provided
    # For now, we rely on the prompt to drive the pipeline
    prompt = request.prompt if request.prompt else request.contract.prompt
    
    try:
        ctx = pipeline.run(prompt)
        
        # Load the final image as base64
        image_path = None
        for file in ctx.run_dir.glob("*.png"):
            if "generated" in file.name:
                image_path = file
                
        img_b64 = None
        if image_path and image_path.exists():
            with open(image_path, "rb") as f:
                img_b64 = base64.b64encode(f.read()).decode("utf-8")
                
        return GenerateResponse(
            run_id=ctx.run_id,
            success=ctx.is_successful,
            verification_report=ctx.verification_report,
            image_base64=img_b64
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("src.api:app", host="0.0.0.0", port=8000, reload=True)

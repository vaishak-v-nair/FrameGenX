from pydantic import BaseModel, Field
from typing import List, Optional, Literal, Dict, Any
import uuid

class BoundingBox(BaseModel):
    xmin: int = Field(..., ge=0)
    ymin: int = Field(..., ge=0)
    xmax: int = Field(..., ge=0)
    ymax: int = Field(..., ge=0)
    
class Entity(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    class_name: str
    attributes: List[str] = Field(default_factory=list)
    relative_position: Optional[Literal["left", "right", "top", "bottom", "center", "foreground", "background"]] = None
    bounding_box: Optional[BoundingBox] = None
    z_index: int = 1

class ParsedIntent(BaseModel):
    subject: Optional[str] = None
    objects: List[str] = Field(default_factory=list)
    colors: List[str] = Field(default_factory=list)
    primary_subject: Optional[str] = None
    secondary_objects: List[str] = Field(default_factory=list)

class Constraint(BaseModel):
    type: str
    value: Any
    target: str
    priority: str

class RagInfluence(BaseModel):
    depth_required: bool = False
    lighting_complex: bool = False

class ExecutionPlan(BaseModel):
    has_human: bool = False
    has_depth: bool = False
    risk_level: str = "medium"
    
class ContractV2(BaseModel):
    schema_version: str = "2.0"
    created_at: str
    prompt: str
    
    # Original Colab "Intelligence" fields
    parsed: ParsedIntent = Field(default_factory=ParsedIntent)
    constraints: List[Constraint] = Field(default_factory=list)
    rag_hits: List[str] = Field(default_factory=list)
    rag_influence: RagInfluence = Field(default_factory=RagInfluence)
    execution_plan: ExecutionPlan = Field(default_factory=ExecutionPlan)
    signal_map: Dict[str, Any] = Field(default_factory=dict)
    
    # New FrameGenX MVP layout engine fields
    entities: List[Entity] = Field(default_factory=list)

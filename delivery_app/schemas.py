from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel


class CustomerCreate(BaseModel):
    name: str
    email: str
    phone: Optional[str] = None
    address: str


class CustomerOut(CustomerCreate):
    id: int
    created_at: datetime

    model_config = {"from_attributes": True}


class StoreCreate(BaseModel):
    name: str
    address: str
    category: Optional[str] = None


class StoreOut(StoreCreate):
    id: int

    model_config = {"from_attributes": True}


class AgentCreate(BaseModel):
    name: str
    phone: str


class AgentOut(AgentCreate):
    id: int
    status: str

    model_config = {"from_attributes": True}


class AgentStatusUpdate(BaseModel):
    status: str


class OrderItemCreate(BaseModel):
    product_name: str
    quantity: int = 1
    unit_price: float


class OrderItemOut(OrderItemCreate):
    id: int
    subtotal: float

    model_config = {"from_attributes": True}


class OrderCreate(BaseModel):
    customer_id: int
    store_id: int
    items: List[OrderItemCreate]
    notes: Optional[str] = None


class OrderOut(BaseModel):
    id: int
    customer_id: int
    store_id: int
    agent_id: Optional[int]
    status: str
    total: float
    delivery_fee: float
    notes: Optional[str]
    created_at: datetime
    items: List[OrderItemOut]

    model_config = {"from_attributes": True}


class StatusUpdate(BaseModel):
    status: str

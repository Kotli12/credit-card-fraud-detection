from datetime import datetime
from typing import List

from fastapi import Depends, FastAPI, HTTPException
from sqlalchemy.orm import Session

from . import models, schemas
from .database import Base, engine, get_db

Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="Delivery App",
    description="Customers order items from stores. Agents shop and deliver.",
    version="1.0.0",
)


# ---------------------------------------------------------------------------
# Customers
# ---------------------------------------------------------------------------


@app.post("/customers", response_model=schemas.CustomerOut, status_code=201, tags=["Customers"])
def create_customer(data: schemas.CustomerCreate, db: Session = Depends(get_db)):
    if db.query(models.Customer).filter(models.Customer.email == data.email).first():
        raise HTTPException(400, "Email already registered")
    customer = models.Customer(**data.model_dump())
    db.add(customer)
    db.commit()
    db.refresh(customer)
    return customer


@app.get("/customers/{customer_id}", response_model=schemas.CustomerOut, tags=["Customers"])
def get_customer(customer_id: int, db: Session = Depends(get_db)):
    customer = db.query(models.Customer).filter(models.Customer.id == customer_id).first()
    if not customer:
        raise HTTPException(404, "Customer not found")
    return customer


@app.get("/customers/{customer_id}/orders", response_model=List[schemas.OrderOut], tags=["Customers"])
def get_customer_orders(customer_id: int, db: Session = Depends(get_db)):
    if not db.query(models.Customer).filter(models.Customer.id == customer_id).first():
        raise HTTPException(404, "Customer not found")
    return db.query(models.Order).filter(models.Order.customer_id == customer_id).all()


# ---------------------------------------------------------------------------
# Stores
# ---------------------------------------------------------------------------


@app.post("/stores", response_model=schemas.StoreOut, status_code=201, tags=["Stores"])
def create_store(data: schemas.StoreCreate, db: Session = Depends(get_db)):
    store = models.Store(**data.model_dump())
    db.add(store)
    db.commit()
    db.refresh(store)
    return store


@app.get("/stores", response_model=List[schemas.StoreOut], tags=["Stores"])
def list_stores(db: Session = Depends(get_db)):
    return db.query(models.Store).all()


@app.get("/stores/{store_id}", response_model=schemas.StoreOut, tags=["Stores"])
def get_store(store_id: int, db: Session = Depends(get_db)):
    store = db.query(models.Store).filter(models.Store.id == store_id).first()
    if not store:
        raise HTTPException(404, "Store not found")
    return store


# ---------------------------------------------------------------------------
# Agents
# ---------------------------------------------------------------------------


@app.post("/agents", response_model=schemas.AgentOut, status_code=201, tags=["Agents"])
def create_agent(data: schemas.AgentCreate, db: Session = Depends(get_db)):
    agent = models.Agent(**data.model_dump())
    db.add(agent)
    db.commit()
    db.refresh(agent)
    return agent


@app.get("/agents", response_model=List[schemas.AgentOut], tags=["Agents"])
def list_agents(db: Session = Depends(get_db)):
    return db.query(models.Agent).all()


@app.put("/agents/{agent_id}/status", tags=["Agents"])
def update_agent_status(agent_id: int, data: schemas.AgentStatusUpdate, db: Session = Depends(get_db)):
    agent = db.query(models.Agent).filter(models.Agent.id == agent_id).first()
    if not agent:
        raise HTTPException(404, "Agent not found")
    valid = [s.value for s in models.AgentStatus]
    if data.status not in valid:
        raise HTTPException(400, f"Status must be one of: {valid}")
    agent.status = data.status
    db.commit()
    return {"message": f"Agent status updated to {data.status}"}


# ---------------------------------------------------------------------------
# Orders
# ---------------------------------------------------------------------------


@app.post("/orders", response_model=schemas.OrderOut, status_code=201, tags=["Orders"])
def create_order(data: schemas.OrderCreate, db: Session = Depends(get_db)):
    if not db.query(models.Customer).filter(models.Customer.id == data.customer_id).first():
        raise HTTPException(404, "Customer not found")
    if not db.query(models.Store).filter(models.Store.id == data.store_id).first():
        raise HTTPException(404, "Store not found")
    if not data.items:
        raise HTTPException(400, "Order must contain at least one item")

    order = models.Order(
        customer_id=data.customer_id,
        store_id=data.store_id,
        notes=data.notes,
        status=models.OrderStatus.pending,
    )
    db.add(order)
    db.flush()

    total = 0.0
    for item_data in data.items:
        item = models.OrderItem(order_id=order.id, **item_data.model_dump())
        db.add(item)
        total += item.quantity * item.unit_price

    order.total = round(total, 2)
    db.commit()
    db.refresh(order)
    return order


@app.get("/orders/{order_id}", response_model=schemas.OrderOut, tags=["Orders"])
def get_order(order_id: int, db: Session = Depends(get_db)):
    order = db.query(models.Order).filter(models.Order.id == order_id).first()
    if not order:
        raise HTTPException(404, "Order not found")
    return order


@app.post("/orders/{order_id}/pay", tags=["Orders"])
def pay_order(order_id: int, db: Session = Depends(get_db)):
    """Simulate payment for a pending order."""
    order = db.query(models.Order).filter(models.Order.id == order_id).first()
    if not order:
        raise HTTPException(404, "Order not found")
    if order.status != models.OrderStatus.pending:
        raise HTTPException(400, f"Order cannot be paid — current status: {order.status}")
    order.status = models.OrderStatus.paid
    order.updated_at = datetime.utcnow()
    db.commit()
    return {
        "message": "Payment successful",
        "order_id": order_id,
        "items_total": order.total,
        "delivery_fee": order.delivery_fee,
        "amount_charged": round(order.total + order.delivery_fee, 2),
    }


@app.post("/orders/{order_id}/assign", tags=["Orders"])
def assign_agent(order_id: int, db: Session = Depends(get_db)):
    """Assign the next available agent to a paid order."""
    order = db.query(models.Order).filter(models.Order.id == order_id).first()
    if not order:
        raise HTTPException(404, "Order not found")
    if order.status != models.OrderStatus.paid:
        raise HTTPException(400, "Order must be paid before an agent can be assigned")

    agent = (
        db.query(models.Agent)
        .filter(models.Agent.status == models.AgentStatus.available)
        .first()
    )
    if not agent:
        raise HTTPException(503, "No agents available right now — please try again shortly")

    order.agent_id = agent.id
    order.status = models.OrderStatus.assigned
    order.updated_at = datetime.utcnow()
    agent.status = models.AgentStatus.busy
    db.commit()
    return {
        "message": f"Agent '{agent.name}' assigned to order #{order_id}",
        "agent_id": agent.id,
        "agent_phone": agent.phone,
    }


_VALID_TRANSITIONS: dict = {
    models.OrderStatus.assigned: [models.OrderStatus.shopping, models.OrderStatus.cancelled],
    models.OrderStatus.shopping: [models.OrderStatus.in_transit],
    models.OrderStatus.in_transit: [models.OrderStatus.delivered],
}


@app.put("/orders/{order_id}/status", tags=["Orders"])
def update_order_status(order_id: int, data: schemas.StatusUpdate, db: Session = Depends(get_db)):
    """
    Agents use this to advance the order through its lifecycle:
      assigned → shopping → in_transit → delivered
    Either step can also be cancelled from assigned.
    """
    order = db.query(models.Order).filter(models.Order.id == order_id).first()
    if not order:
        raise HTTPException(404, "Order not found")

    try:
        new_status = models.OrderStatus(data.status)
    except ValueError:
        valid = [s.value for s in models.OrderStatus]
        raise HTTPException(400, f"Invalid status '{data.status}'. Must be one of: {valid}")

    current = models.OrderStatus(order.status)
    allowed = _VALID_TRANSITIONS.get(current, [])
    if new_status not in allowed:
        allowed_names = [s.value for s in allowed] if allowed else []
        raise HTTPException(
            400,
            f"Cannot move from '{current}' to '{new_status}'. "
            f"Allowed next statuses: {allowed_names or 'none'}",
        )

    order.status = new_status
    order.updated_at = datetime.utcnow()

    if new_status in (models.OrderStatus.delivered, models.OrderStatus.cancelled):
        if order.agent_id:
            agent = db.query(models.Agent).filter(models.Agent.id == order.agent_id).first()
            if agent:
                agent.status = models.AgentStatus.available

    db.commit()
    return {"message": f"Order #{order_id} status updated to '{new_status}'"}

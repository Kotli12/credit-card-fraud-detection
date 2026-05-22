"""Entry point: python run_delivery.py"""
import uvicorn

if __name__ == "__main__":
    uvicorn.run("delivery_app.main:app", host="0.0.0.0", port=8000, reload=True)

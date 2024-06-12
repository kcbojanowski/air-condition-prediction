import asyncio
import click
import httpx
from rich.console import Console
from datetime import datetime, timedelta
import random
import time
import json
import websockets

console = Console()
API_URL = "http://localhost:8000/api"

@click.group()
def cli():
    pass

async def generate_air_quality_data():
    base_date = datetime.now() - timedelta(days=30)
    
    async with websockets.connect("ws://localhost:8000/ws") as websocket:
        start_time = time.time()
        while time.time() - start_time < 15:
            current_date = base_date + timedelta(days=random.randint(0, 30))
            pm10_value = round(random.uniform(10, 150), 2)
            
            data = {
                "timestamp": current_date.strftime("%Y-%m-%d %H:%M:%S"),
                "pm10": pm10_value
            }
            
            await websocket.send(json.dumps(data))
            await asyncio.sleep(1)  # symuluje generowanie danych co sekundę

@cli.command()
def train_model():
    """Trigger model training."""
    response = httpx.post(f"{API_URL}/build-and-train")
    console.print(response.json())

@cli.command()
def evaluate_model():
    """Evaluate the model."""
    response = httpx.post(f"{API_URL}/evaluate")
    console.print(response.json())

@cli.command()
def generate_data():
    """Generate air quality data."""
    asyncio.run(generate_air_quality_data())
    console.print("Data generation completed.")


if __name__ == "__main__":
    cli()
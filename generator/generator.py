import asyncio
import click
import httpx
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeRemainingColumn
from datetime import datetime, timedelta
import random
import time
import json
import websockets

console = Console()
API_URL = "http://backend:8000/api"
TIMEOUT = 50.0


sent_data = []

@click.group()
def cli():
    pass

async def generate_air_quality_data():
    async with websockets.connect("ws://backend:8000/ws") as websocket:
        start_time = time.time()
        while time.time() - start_time < 15:
            pm10_value = round(random.uniform(10, 150), 2)
            
            data = {
                "pm10": pm10_value
            }

            sent_data.append(data["pm10"])
            
            await websocket.send(json.dumps(data))
            await asyncio.sleep(1)
                
                
@cli.command()
def train_model():
    """Trigger model training."""
    with Progress(SpinnerColumn(), TextColumn("[bold green]Training model...")) as progress:
        task = progress.add_task("")
        
        try:
            response = httpx.post(f"{API_URL}/build-and-train", timeout=TIMEOUT)
            response_json = response.json()
            progress.stop()
            
            console.print(f"{response_json['status']}.")
            
            table = Table(title="Training metrics")
            table.add_column("Training Time (seconds)", justify="right", style="cyan", no_wrap=True)
            table.add_column("Train Loss", style="magenta")
            
            table.add_row(f"{round(response_json['time_taken'],  2)}", f"{response_json['train_loss']}")
            
            console.print('\n')
            console.print(table)

        except httpx.TimeoutException:
            progress.stop()
            console.print("The request timed out.")
        except httpx.RequestError as exc:
            progress.stop()
            console.print(f"An error occurred while requesting {exc.request.url!r}.")
        except httpx.HTTPStatusError as exc:
            progress.stop()
            console.print(f"Error response {exc.response.status_code} while requesting {exc.request.url!r}.")
        

@cli.command()
def evaluate_model():
    """Evaluate the model."""
    try:
        response = httpx.post(f"{API_URL}/evaluate", timeout=TIMEOUT)
        response_json = response.json()
        
        table = Table(title="Model Evaluation Metrics")
        table.add_column("Metric", justify="right", style="cyan", no_wrap=True)
        table.add_column("Value", style="magenta")

        table.add_row("MEA (Mean Absolute Error)", f"{response_json['mae']}")
        table.add_row("MSE (Mean Squared Error)", f"{response_json['mse']}")
        table.add_row("RMSE (Root Mean Squared Error)", f"{response_json['rmse']}")

        console.print()
        console.print(table)
        
    except httpx.TimeoutException:
        console.print("The request timed out.")
    except httpx.RequestError as exc:
        console.print(f"An error occurred while requesting {exc.request.url!r}.")
    except httpx.HTTPStatusError as exc:
        console.print(f"Error response {exc.response.status_code} while requesting {exc.request.url!r}.")
    


@cli.command()
def generate_data():
    """Generate air quality data."""
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold green]Generating air quality data..."),
        BarColumn(),
        TimeRemainingColumn()
    ) as progress:
        task = progress.add_task("Generating data", total=15)
        asyncio.run(generate_air_quality_data())
        progress.update(task, completed=15)
        
    console.print("Data generation completed.")
    
    table = Table(title="Generated Air Condition Data")
    table.add_column("Date", justify="right", style="cyan", no_wrap=True)
    table.add_column("Air Condition", style="magenta")
    
    start_date = datetime.today() - timedelta(days=14)
    
    for i in range(len(sent_data)):
        date = start_date + timedelta(days=i)
        formatted_date = date.strftime("%Y-%m-%d")
        condition = str(sent_data[i])
        table.add_row(formatted_date, condition)
    
    console.print('\n')
    console.print(table)

@cli.command()
def get_predictions():
    """Get air quality predictions for the next 5 days"""
    response = httpx.post(f"{API_URL}/get-predictions", timeout=TIMEOUT)
    response_data = response.json()
    
    table = Table(title=response_data["text"])
    table.add_column("Day", justify="right", style="cyan", no_wrap=True)
    table.add_column("Prediction", style="magenta")
    today = datetime.today()
    
    for i in range(1, 6):
        date = today + timedelta(days=i)
        formatted_date = date.strftime("%Y-%m-%d")
        prediction = round(response_data["data"][i], 2)
        table.add_row(formatted_date, str(prediction))
    
    console.print('\n')
    console.print(table)


if __name__ == "__main__":
    cli()
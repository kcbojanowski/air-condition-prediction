# import asyncio
# import docker
# from pathlib import Path
# from PyInquirer import prompt, Separator
# from scripts.generator import generate_air_quality_data
# from rich.console import Console

# console = Console()

# def run_docker_container(image_name: str, ports: dict):
#     client = docker.from_env()
#     container = client.containers.run(image_name, ports=ports, detach=True)
#     return container

# def stop_docker_container(container):
#     container.stop()
#     container.remove()

# def main_menu():
#     questions = [
#         {
#             'type': 'list',
#             'name': 'action',
#             'message': 'Select an action:',
#             'choices': [
#                 'Train Model',
#                 'Evaluate Model',
#                 'Generate Data',
#                 'Run Server',
#                 'Exit'
#             ]
#         }
#     ]
#     answers = prompt(questions)
#     return answers['action']

# def runserver():
#     """Run the FastAPI server in Docker."""
#     console.print("Starting FastAPI server...")
#     container = run_docker_container("air_quality_app", {"8000/tcp": 8000})
#     console.print(f"Server running at http://localhost:8000")
#     console.print("Press Ctrl+C to stop the server.")
#     try:
#         container.wait()
#     except KeyboardInterrupt:
#         console.print("Stopping the server...")
#         stop_docker_container(container)

# def train():
#     """Trigger model training."""
#     from app.core.train_model import train_model
#     train_model()
#     console.print("Training started.")

# def evaluate():
#     """Evaluate the model."""
#     import torch
#     import numpy as np
#     from app.core.model import ModelInstance
#     from app.core.data_processing import normalization, create_dataset

#     data = np.random.rand(1000)
#     data_normalized = normalization(data)
#     lookback = 3
#     X, y = create_dataset(data_normalized, lookback)
#     X = torch.tensor(X, dtype=torch.float32)
#     y = torch.tensor(y, dtype=torch.float32)

#     model_instance = ModelInstance()
#     metrics = model_instance.evaluate_model(X, y)
#     console.print(metrics)

# def generate_data():
#     """Generate air quality data."""
#     asyncio.run(generate_air_quality_data())
#     console.print("Data generation completed.")

# def main():
#     console.print("[bold cyan]Welcome to the Air Quality CLI![/bold cyan]")
#     while True:
#         action = main_menu()
#         if action == 'Train Model':
#             train()
#         elif action == 'Evaluate Model':
#             evaluate()
#         elif action == 'Generate Data':
#             generate_data()
#         elif action == 'Run Server':
#             runserver()
#         elif action == 'Exit':
#             console.print("Exiting...")
#             break

# if __name__ == "__main__":
#     main()
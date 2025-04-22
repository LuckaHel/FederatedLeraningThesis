# runAllClients.py
import subprocess
import time

def run_server_and_clients(server_file, start_client, end_client):
    print("🚀 Starting server...")
    server_process = subprocess.Popen(["python3", server_file])
    time.sleep(3)  # Wait a moment before starting clients

    client_processes = []
    for i in range(start_client, end_client + 1):
        client_file = f"bertClient{i}_gpu.py"
        print(f"⚙️  Launching {client_file}...")
        proc = subprocess.Popen(["python3", client_file])
        client_processes.append(proc)
        time.sleep(1)  # Optional: stagger startup

    # Wait for all clients to finish
    for proc in client_processes:
        proc.wait()

    server_process.wait()

if __name__ == "__main__":
    run_server_and_clients(server_file="bertserver_gpu.py", start_client=1, end_client=2)

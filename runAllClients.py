import subprocess
import time

def run_server_and_clients(server_script, client_start, client_end):
    # Start the server
    server_process = subprocess.Popen(["python", server_script])
    time.sleep(2)  # Give the server time to start

    # Start clients
    client_processes = []
    for i in range(client_start, client_end + 1):
        client_script = f"client{i}.py"
        client_process = subprocess.Popen(["python", client_script])
        client_processes.append(client_process)
        time.sleep(1)  # Optional: Stagger the client starts

    # Wait for all clients to finish
    for client_process in client_processes:
        client_process.wait()

    # Stop the server
    server_process.terminate()
    print("Server and all clients have finished.")

# Usage
run_server_and_clients("server1.py", 4, 9)


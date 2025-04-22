# copyClients.py
import shutil

# Set range of clients you want to create
start_client = 2
end_client = 2
template_file = "bertClient1_gpu.py"

for i in range(start_client, end_client + 1):
    target_file = f"bertClient{i}_gpu.py"
    shutil.copy(template_file, target_file)
    print(f"✅ Created: {target_file}")

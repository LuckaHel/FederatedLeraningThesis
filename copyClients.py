import shutil

def create_multiple_clients(source_file, start_num, end_num):
    for i in range(start_num, end_num + 1):
        new_file = f"client{i}.py"
        shutil.copy(source_file, new_file)
        print(f"Created {new_file}")

# Usage
create_multiple_clients("client1.py", 4, 100)


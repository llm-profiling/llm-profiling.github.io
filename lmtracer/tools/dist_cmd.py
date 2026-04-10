
import subprocess


def remote_exec(command, user, host, port=22):

    ssh_command = ["ssh", f"-p{port}", f"{user}@{host}", command]
    result = subprocess.run(ssh_command, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"Command failed on {user}@{host}:{port} with error: {result.stderr}")

    return result.stdout.strip()

def local_exec(command):

    result = subprocess.run(command, shell=True, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"Local command failed with error: {result.stderr}")

    return result.stdout.strip()
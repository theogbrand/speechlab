import modal

stub = modal.Stub("abax-asr")

server_image = modal.Image.from_dockerfile(
    "Dockerfile.server",
)

worker_image = modal.Image.from_dockerfile(
    "Dockerfile.worker",
)


@stub.function(image=server_image)
def run_server():
    import subprocess

    subprocess.run(["/home/speechuser/start_master.sh", "-p", "8010"])


@stub.function(image=worker_image, gpu="any")
def run_worker():
    import subprocess

    subprocess.run(
        ["/home/speechuser/start_worker.sh", "-m", "decoding-sdk-server", "-p", "8080"]
    )


@stub.local_entrypoint()
def main():
    server = run_server.spawn()
    worker = run_worker.spawn()

    print("Server and worker are running. Press Ctrl+C to stop.")

    try:
        server.get()
        worker.get()
    except KeyboardInterrupt:
        print("Stopping server and worker...")
    finally:
        modal.Function.shutdown_all()


if __name__ == "__main__":
    modal.runner.main(main)

# GPU access for Docker containers (verified 2026-11-18)

Goal: let the project's containerized Ollama run on the NVIDIA GPU.

## Verified machine

| Item | Value |
|---|---|
| OS | Fedora 44, SELinux **enforcing** |
| GPU | NVIDIA GeForce RTX 5050 Laptop GPU, 8 GB, compute capability 12.0 (Blackwell) |
| Driver | 620.70.06 (RPM Fusion `akmod-nvidia`), CUDA UMD 13.5 |
| Docker | docker-ce 29.10.0, Compose v5.7.1, CDI enabled by default (`/etc/cdi`, `/var/run/cdi`) |
| Toolkit | nvidia-container-toolkit 1.20.2 (from NVIDIA's repo) |
| Ollama image | `ollama/ollama:latest` = 0.40.16, uses bundled `cuda_v13` libs |

## Method: CDI (Container Device Interface)

No `daemon.json` change and no extra Docker runtime are needed. SELinux stays enforcing and needs **no** label workaround (`label=disable` / `container_use_devices` were not required).

One-time host setup (needs sudo):

```bash
curl -s -L https://nvidia.github.io/libnvidia-container/stable/rpm/nvidia-container-toolkit.repo | sudo tee /etc/yum.repos.d/nvidia-container-toolkit.repo
```
```bash
sudo dnf install -y nvidia-container-toolkit
```
```bash
sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml
```
```bash
nvidia-ctk cdi list   # expect nvidia.com/gpu=all
```

Ubuntu/Debian hosts: install the toolkit from the same NVIDIA repo with apt (see NVIDIA's install guide), then run the same `cdi generate` command.

**Re-run `nvidia-ctk cdi generate` after every NVIDIA driver update**, otherwise containers fail to start with a CDI/library mismatch.

## Compose snippet (verified)

```yaml
services:
  ollama:
    image: ollama/ollama:latest
    devices:
      - nvidia.com/gpu=all          # CDI device name
```

Equivalent long form (also verified):

```yaml
    deploy:
      resources:
        reservations:
          devices:
            - driver: cdi
              device_ids: ["nvidia.com/gpu=all"]
              capabilities: [gpu]
```

Plain `docker run` equivalent: `docker run --device nvidia.com/gpu=all ...`

## What does NOT work here (gotchas)

- `docker run --gpus all` → `Error response from daemon: AMD CDI spec not found`.
- Compose `deploy.resources.reservations.devices: [{driver: nvidia, count: all, capabilities: [gpu]}]` (the syntax in Ollama's/Docker's docs) → `could not select device driver "nvidia"`.
  Both need the legacy NVIDIA runtime (`sudo nvidia-ctk runtime configure --runtime=docker` + docker restart), which we deliberately did **not** configure. Hosts that set up only the legacy runtime (no CDI spec) need that syntax instead; CDI is the portable choice for this repo.
- 8 GB VRAM cannot hold llama3.1:8b twice. Before running the containerized Ollama with the 8B model, free the GPU from the host service: `sudo systemctl stop ollama` and afterwards `sudo systemctl start ollama` to get it back. (The test below ran alongside the idle host service without conflict because only the 0.5B model was loaded.)
- Don't publish the container's port as `11434` on the host — the host Ollama service already binds `127.0.0.1:11434`. Keep it internal to the compose network, or map to another port (e.g. `127.0.0.1:11435:11434`).
- The container cannot reuse the host's models (`/usr/share/ollama/.ollama/models` is owned by the `ollama` user); use a named volume at `/root/.ollama` and pull models into it.
- `ollama/ollama:latest` image is ~8.5 GB on disk (already pulled on this machine).

## Verification evidence

1. `docker run --rm --device nvidia.com/gpu=all ubuntu:24.04 nvidia-smi` → shows RTX 5050, driver 620.70.06.
2. `ollama/ollama` container with `--device nvidia.com/gpu=all`, port `127.0.0.1:11435`, throwaway volume:
   - startup log: `inference compute ... library=CUDA compute=12.0 ... "NVIDIA GeForce RTX 5050 Laptop GPU" libdirs=ollama,cuda_v13 ... total="7.6 GiB"` → Blackwell is supported by the image.
   - `qwen2.5:0.5b` load: `offloaded 25/25 layers to GPU`; `/api/ps`: `size == size_vram` (100% GPU).
   - generation speed: ~262 tokens/s; host `nvidia-smi` showed the container's `llama-server` using 558 MiB.
3. Throwaway container, volume and the `ubuntu:24.04` test image were removed afterwards.

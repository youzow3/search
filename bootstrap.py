
import argparse
import os
import subprocess

def check_command(command: str) -> int:
    print(f"Checking \"{command}\" is available...")
    try:
        r: subprocess.CompletedProcess = subprocess.run([command, "--version"])
    except:
        print(f"Command \"{command}\" cannot run. Please check you have installed it or set PATH properly.")
        return 1
    print("Done")
    return 0

def get_llama_command_path(command: str, config: str) -> str:
    cmd: str

    cmd = f"llama.cpp/Build/bin/{command}"
    if check_command(cmd) == 0:
        return cmd

    cmd = f"llama.cpp/Build/bin/{config}/{command}"
    if check_command(cmd) == 0:
        return cmd
    return None

def make_server_script(config: str, server_args: str, thread: int, model_path: str) -> int:
    if config is None:
        config = "Debug"

    command: str = get_llama_command_path("llama-server", config)
    if command is None:
        print("Failed to find llama-server binary. Please make the server script manually.")
        return 1

    server_args_split: list[str] = server_args.split()
    if not "-t" in server_args_split or not "--thread" in server_args_split:
        server_args += f" -t {thread}"
    server_command: str = f"{command} {server_args} -m {model_path}"

    with open("llama-server.sh", "w") as f:
        f.writelines(["#!/bin/sh", server_command])
    with open("llama-server.bat", "w") as f:
        _ = [print(s, file = f) for s in server_command])
    return 0

def download_model(huggingface: str, gguf: str) -> int:
    returncode: int
    if huggingface is not None:
        directory: str = huggingface.split('/')[-1].removesuffix(".git")
        os.system("git lfs install")
        returncode = os.system(f"git clone {huggingface}")
        if returncode != 0:
            returncode = os.system(f"git -C {directory} pull")
            if returncode != 0:
                return 1
        returncode = os.system(f"python3 llama.cpp/convert_hf_to_gguf.py --outfile model.gguf {directory}")
    else:
        returncode = os.system("wget -O model.gguf {gguf}")

    return returncode

def quantize_model(config: str, quant_type: str, thread: int):
    quantize: str = get_llama_command_path("llama-quantize", config)
    returncode: int = os.system(f"{quantize} model.gguf model-{quant_type}.gguf {quant_type} {thread}")
    return returncode

def main(args: argparse.Namespace) -> int:
    check: int = sum([check_command(cmd) for cmd in ["git", "git-lfs", "cmake"]])
    if check != 0:
        print("Some necessary softwares were not found. Please check them, and try again.")
        return 1

    returncode: int

    print("Start building llama.cpp")
    returncode = os.system("git clone https://github.com/ggerganov/llama.cpp.git")
    if returncode != 0:
        returncode = os.system("git -C llama.cpp pull")
        if returncode != 0:
            print("Failed to clone llama.cpp")
            return 1

    if args.revision != "master":
        returncode = os.system(f"git checkout {args.revision}")
        if returncode != 0:
            print("Warning: Failed to checkout {args.revision}")

    os.system("pip install -r llama.cpp/requirements.txt")

    cmake_configure: int = os.system("cmake llama.cpp -B llama.cpp/Build " + args.build_config)
    if cmake_configure != 0:
        print("Failed to configure llama.cpp")
        return 1

    build_command: str = "cmake --build llama.cpp/Build"
    if args.config is not None:
        build_command += f" --config {args.config}"
    if args.thread is not None:
        build_command += f" -j{args.thread}"
    returncode = os.system(build_command)
    if returncode != 0:
        print("Failed to build llama.cpp")
        return 1
    print("Done")

    print("Start downloading AI model.")
    returncode = download_model(args.model_huggingface, args.model_gguf)
    if returncode != 0:
        print("Failed to download the LLM files.")
        return 1
    print("Done")

    if args.quantize is not None:
        print("Start quantizing")
        returncode = quantize_model(args.config, args.quantize, args.thread)
        if returncode != 0:
            print("Warning: Failed to quantize the GGUF file.")
        print("Done")

    print("Start making the server script.")
    model_path: str = "model.gguf" if args.quantize is None else f"model-{args.quantize}"
    returncode = make_server_script(args.config, args.server_argument, args.thread, model_path)
    if returncode != 0:
        print("Warning: Failed to make server script.")
    print("Done")

    return 0

if __name__ == "__main__":
    parser: argparse.ArgumentParser = argparse.ArgumentParser()
    parser.add_argument("--build-config", default = "", help = "Cmake configuration arguments")
    parser.add_argument("--config", default = None, help = "Build configuration for llama.cpp. Maybe \"Debug\" or \"Release\".")
    parser.add_argument("--revision", default = "master", help = "Revision for llama.cpp")
    parser.add_argument("-t", "--thread", default = 1, help = "Parallel build and quantization", type = int)
    parser.add_argument("--model-huggingface", default = None, help = "Git URL for the huggingface model.")
    parser.add_argument("--model-gguf", default = "https://huggingface.co/Qwen/Qwen2-7B-Instruct-GGUF/resolve/main/qwen2-7b-instruct-fp16.gguf", help = "Link for the GGUF model.")
    parser.add_argument("--quantize", default = None, help = "Quantize type")
    parser.add_argument("--server-argument", default = "-v -fa -t 8 -c 0 -ngl 4096 --log-append", help = "Argument for llama-server. You don't have to specify models using '-m' or '--model'.")
    exit(main(parser.parse_args()))

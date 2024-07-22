
import argparse
import logging
import os
import subprocess
import sys

from typing import TextIO

logging.basicConfig(format="%(levelname)s: %(message)s", level = logging.DEBUG)
__logger = logging.getLogger(__name__)

def check_command(command: str) -> int:

    __logger.info(f"Checking \"{command}\"")
    try:
        r: subprocess.CompletedProcess = subprocess.run([command, "--version"], stdout = subprocess.DEVNULL, stderr = subprocess.DEVNULL)
    except:
        __logger.error(f"The command \"{command}\" not found")
        return 1
    __logger.info(f"The command \"{command} found.")
    return 0

def get_llama_command_path(config: str) -> str:
    if check_command("./llama.cpp/Build/bin/llama-cli") == 0:
        return f"./llama.cpp/Build/bin/"
    elif check_command("./llama.cpp/Build/bin/{config}/llama-cli") == 0:
        return f"./llama.cpp/Build/bin/{config}/"
    return None

def make_server_script(llama_bin: str, server_args: str, thread: int, model_path: str) -> int:
    __logger.info("Start making llama-server script")

    server_args_split: list[str] = server_args.split()
    if not "-t" in server_args_split or not "--thread" in server_args_split:
        server_args += f" -t {thread}"
    server_command: str = f"{llama_bin}llama-server {server_args} -m {model_path}"

    with open("llama-server.sh", "w") as f:
        _ = [print(s, file = f) for s in ["#!/bin/sh", server_command]]
    with open("llama-server.bat", "w") as f:
        print(server_command, file = f)

    __logger.info("Finished Making llama-server script")
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
        returncode = os.system(f"wget -O model.gguf {gguf}")

    return returncode

def quantize_model(llama_bin: str, quant_type: str, thread: int):
    returncode: int = os.system(f"{llama_bin}llama-quantize model.gguf model-{quant_type}.gguf {quant_type} {thread}")
    return returncode

def main(args: argparse.Namespace) -> int:
    check: int = sum([check_command(cmd) for cmd in ["git", "git-lfs", "cmake", "wget", "pip"]])
    if check != 0:
        __logger.critical("Some necessary softwares were not found. Please check them, and try again.")
        return 1

    returncode: int

    __logger.info("Start building llama.cpp")
    returncode = os.system("git clone https://github.com/ggerganov/llama.cpp.git")
    if returncode != 0:
        returncode = os.system("git -C llama.cpp pull")
        if returncode != 0:
            __logger.critical("Failed to clone or pull llama.cpp")
            return 1

    if args.revision != "master":
        returncode = os.system(f"git checkout {args.revision}")
        if returncode != 0:
            __logger.warning("Warning: Failed to checkout {args.revision}")

    returncode = os.system("pip install -r llama.cpp/requirements.txt")
    if returncode != 0:
        __logger.critical("Failed to install required modules.")
        return 1

    returncode = os.system("cmake llama.cpp -B llama.cpp/Build " + args.build_config)
    if returncode != 0:
        __logger.critical("Failed to configure llama.cpp")
        return 1

    build_command: str = "cmake --build llama.cpp/Build"
    if args.config is not None:
        build_command += f" --config {args.config}"
    if args.thread is not None:
        build_command += f" -j{args.thread}"
    returncode = os.system(build_command)
    if returncode != 0:
        __logger.critical("Failed to build llama.cpp")
        return 1

    llama_bin: str = get_llama_command_path(args.config)
    if llama_bin is None:
        __logger.warning("Failed to locate binaries directory. Quantizing and generating server script will not be run")
    __logger.info("Finished building llama.cpp")

    if not args.model_skip:
        __logger.info("Start downloading AI model")
        returncode = download_model(args.model_huggingface, args.model_gguf)
        if returncode != 0:
            __logger.error("Failed to download the LLM files.")
        else:
            __logger.info("Finished downloading AI model")

    if args.quantize is not None and llama_bin is not None:
        __logger.info("Start quantizing the model")
        returncode = quantize_model(llama_bin, args.quantize, args.thread)
        if returncode != 0:
            __logger.warning("Warning: Failed to quantize the GGUF file.")
        else:
            __logger.info("Finished quantizing the model")

    model_path: str = "model.gguf" if args.quantize is None else f"model-{args.quantize}.gguf"
    if llama_bin is not None:
        make_server_script(llama_bin, args.server_argument, args.thread, model_path)

    return 0

if __name__ == "__main__":
    parser: argparse.ArgumentParser = argparse.ArgumentParser()
    parser.add_argument("--build-config", default = "", help = "Cmake configuration arguments")
    parser.add_argument("--config", default = None, help = "Build configuration for llama.cpp. Maybe \"Debug\" or \"Release\".")
    parser.add_argument("--revision", default = "master", help = "Revision for llama.cpp")
    parser.add_argument("-t", "--thread", default = 1, help = "Parallel build and quantization", type = int)
    parser.add_argument("--model-huggingface", default = None, help = "Git URL for the huggingface model.")
    parser.add_argument("--model-gguf", default = "https://huggingface.co/Qwen/Qwen2-7B-Instruct-GGUF/resolve/main/qwen2-7b-instruct-fp16.gguf", help = "Link for the GGUF model.")
    parser.add_argument("--model-skip", action="store_true", default = False, help = "Skip downloading AI model.")
    parser.add_argument("--quantize", default = None, help = "Quantize type")
    parser.add_argument("--server-argument", default = "-v -fa -t 8 -c 0 -ngl 4096 --log-append", help = "Argument for llama-server. You don't have to specify models using '-m' or '--model'.")
    exit(main(parser.parse_args()))

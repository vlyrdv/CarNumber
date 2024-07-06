import sys
from subprocess import call


def run_cmd(command):
    try:
        print(command)
        call(command, shell=True)
    except KeyboardInterrupt:
        print("Process interrupted")
        sys.exit(1)


def start_improvment():
    INPUT_DIR = 'CarNumbers/utils/Real-ESRGAN/inputs'
    OUTPUT_DIR = 'CarNumbers/utils/Real-ESRGAN/results'
    run_cmd(
        "python CarNumbers/utils/Real-ESRGAN/inference_realesrgan.py --model_path CarNumbers/utils/Real-ESRGAN/experiments/pretrained_models/RealESRGAN_x4plus.pth --input " + INPUT_DIR + " --output " + OUTPUT_DIR + " --outscale 3 --fp32 ")





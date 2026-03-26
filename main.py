"""
MIT Licence

Copyright © 2026 DreamOfTommorowWorld

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without
restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

from diffusers import DiffusionPipeline
import torch
import imageio

print("""
=== WARNING!!! ===
Running a AI model locally is resource intensive (usually) and also, AI models are QUITE HEAVY so if
you have ample amount of storage and have good enough hardware! And yeah, AI consumes A LOT OF
ELECTRICITY (check: How much SORA AI consumes electricity and how much carbon footprint it leaves)!

=== Computer stuff you'd need! ===
+++ Hardware +++
(you need to buy them)
- A Nvidia GPU with CUDA installed
+++ Software +++
(go to python.org to install them)
- Python 3.10 (to Python 3.13)
+++ Python libraries +++
(run 'pip install torch diffusers transformers accelerate imageio imageio[ffmpeg] imageio[pyav]' in your terminal to install them)
- PyTorch
- HuggingFace diffusers
- HuggingFace Transformers
- imageio python library
- imageio

IF YOU FIND THAT SOMETHING IS BROKEN, PLEASE GIVE ME A PULL REQUEST ON THE GITHUB REPO! And yeah, suggestions are welcome too!

This thing is under active development!
""")

confirm = input("Are you sure to continue??? (Y/N): ")
if confirm == "y" or confirm == "Y":
    print("Okay!")
    pass
else:
    print("QUITTING...")
    exit()
    
# Load Hugging Face text-to-video model
pipe = DiffusionPipeline.from_pretrained(
    "damo-vilab/text-to-video-ms-1.7b", 
    torch_dtype=torch.float16, 
    variant="fp16"
).to("cuda")  # Use GPU for speed
prompt = input("Enter your prompt: ")

# Generate frames
video_frames = pipe(prompt, num_frames=16).frames

# Save frames to MP4
output_path = input("Enter the filename to save the AI-generated video (.mp4 is recommended!): ")
imageio.mimsave(output_path, video_frames, fps=8)

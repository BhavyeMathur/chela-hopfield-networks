from PIL import Image
import os

input_folder = "output"
output_file = "output.gif"
fps = 30

images = sorted([img for img in os.listdir(input_folder) if img.endswith(".png")], key=lambda x: int(x[:-4]))

frames = []
for image in images:
    img_path = os.path.join(input_folder, image)
    frame = Image.open(img_path)
    frame = frame.resize((512, 512), Image.NEAREST)
    frames.append(frame)

frames[0].save(output_file, save_all=True, append_images=frames[1:], duration=1000 // fps, loop=0, optimize=True)
print(f"GIF saved as {output_file}")

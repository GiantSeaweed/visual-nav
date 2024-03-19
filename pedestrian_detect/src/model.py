import requests
from PIL import Image
import torch
from PIL import ImageDraw
from transformers import OwlViTProcessor, OwlViTForObjectDetection

processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32")

for i in range(1580):
    # id has four digits padding with 0
    id = str(i).zfill(4)
    url = f"/local2/wu1827_/robot/558rl/video_frames/frame{id}.jpg"
    image = Image.open(url) # Image.open(requests.get(url, stream=True).raw)

    # img_path = "/local2/wu1827_/robot/558rl/video_frames/frame1391.jpg"
    # image = Image.open(img_path)

    texts = [["a photo of pedestrians"]]
    inputs = processor(text=texts, images=image, return_tensors="pt")
    outputs = model(**inputs)

    # Target image sizes (height, width) to rescale box predictions [batch_size, 2]
    target_sizes = torch.Tensor([image.size[::-1]])
    # Convert outputs (bounding boxes and class logits) to Pascal VOC format (xmin, ymin, xmax, ymax)
    results = processor.post_process_object_detection(outputs=outputs, target_sizes=target_sizes, threshold=0.1)
    i = 0  # Retrieve predictions for the first image for the corresponding text queries
    text = texts[i]
    boxes, scores, labels = results[i]["boxes"], results[i]["scores"], results[i]["labels"]
    # image =  Image.open( url)
    for box, score, label in zip(boxes, scores, labels):
        box = [round(i, 2) for i in box.tolist()]
        # print(f"Detected {text[label]} with confidence {round(score.item(), 3)} at location {box}")
        # save image with all the boxes
        draw = ImageDraw.Draw(image)
        draw.rectangle(box, outline="red", width=3)
        # image.show()


    image.save(f"/local2/wu1827_/robot/558rl/test/test-large{id}.jpg")
    print(f"frame{id}.jpg is processed")



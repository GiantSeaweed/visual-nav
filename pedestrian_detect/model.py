import requests
from PIL import Image
import torch
from PIL import ImageDraw
from transformers import OwlViTProcessor, OwlViTForObjectDetection
import json
from ultralytics import SAM
from segment_anything import SamPredictor, sam_model_registry
from huggingface_hub import hf_hub_download
import cv2
import supervision as sv
import numpy as np
import matplotlib.pyplot as plt

def get_bounding_box():
    processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
    model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32")

    for i in [1138]: #range(1580):
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
        # save box to 
        save_json =f"/local2/wu1827_/robot/558rl/box_outputs/boxes{id}.json"
        with open(save_json, 'w') as f:
            json.dump(boxes.tolist(), f)
        # break
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


def run_mobileSAM():
    

    # Load the model
    model = SAM('mobile_sam.pt')

    # Predict a segment based on a box prompt
    results= model.predict('/local2/wu1827_/robot/558rl/test/test0000.jpg', bboxes=[1054.898681640625, 43.06448745727539, 1194.3568115234375, 284.0183410644531])
    # print(output)
    # exit()
    print("type(results)", type(results))#list
    print("len(results)", len(results))#1
    print("type(results[0])", type(results[0]))
    for r in results:
        im_array = r.plot()  # plot a BGR numpy array of predictions
        im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image
        # im.show()  # show image
        im.save('/local2/wu1827_/robot/558rl/mobile_sam/results.jpg')  # save image
    # exit()
    # # save output to /local2/wu1827_/robot/558rl/mobile_sam
    # # with open('/local2/wu1827_/robot/558rl/mobile_sam/output0000.json', 'w') as f:
    # #     json.dump(output, f)
    # # apply results on the image
    # image = Image.open('/local2/wu1827_/robot/558rl/test/test0000.jpg')
    # draw = ImageDraw.Draw(image)
    # for box in output['bboxes']:
    #     draw.rectangle(box, outline="red", width=3)
    # image.save('/local2/wu1827_/robot/558rl/mobile_sam/output0000.jpg')

def show_mask(mask, ax):
    ax.imshow(mask, alpha=0.5, cmap="viridis")

def show_box(box, ax):
    ax.add_patch(plt.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], fill=False, color="red"))

def combine_masks(all_masks, shape):
    """Combine all masks into one binary mask."""
    combined_mask = np.zeros(shape[:2], dtype=np.uint8)  # Assuming all_masks are binary
    for mask in all_masks:
        combined_mask = np.logical_or(combined_mask, mask > 0).astype(np.uint8)
    return combined_mask

def show_mask_on_image(image, combined_mask, ax):
    """Show the image with the mask overlaid."""
    # Convert the mask to a color (RGBA) image
    mask_color = np.zeros((combined_mask.shape[0], combined_mask.shape[1], 4), dtype=np.uint8)
    mask_color[combined_mask > 0] = [0, 255, 0, 100]  # gree with transparency
    
    # Convert the original image to RGBA
    image_rgba = cv2.cvtColor(image, cv2.COLOR_RGB2RGBA)
    
    # Overlay the mask on the image
    overlayed_image = cv2.addWeighted(image_rgba, 1, mask_color, 0.5, 0)
    
    # Show the result
    ax.imshow(cv2.cvtColor(overlayed_image, cv2.COLOR_RGBA2RGB))

    
def sam():


    # chkpt_path = hf_hub_download( "checkpoints/sam_vit_b_01ec64.pth")       
    # print("chkpt_path", chkpt_path)
    # exit()
    DEVICE = torch.device('cuda:5' if torch.cuda.is_available() else 'cpu')

    MODEL_TYPE = "vit_l"

    sam = sam_model_registry[MODEL_TYPE](checkpoint="/local2/wu1827_/robot/558rl/sam_vit_l_0b3195.pth")
    sam.to(device=DEVICE)


    mask_predictor = SamPredictor(sam)
    IMAGE_PATH  = "/local2/wu1827_/robot/558rl/test/test1138.jpg"
    image_bgr = cv2.imread(IMAGE_PATH)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    mask_predictor.set_image(image_rgb)
    box_file = "/local2/wu1827_/robot/558rl/box_outputs/boxes1138.json"
    with open(box_file, 'r') as f:
        box_list= json.load(f)

    # box = np.array([70, 247, 626, 926])
    all_masks = []
    # predict the mask for each box
    

    # draw all the mask on the image
    for box in box_list:
        #box = np.array([1054.898681640625, 43.06448745727539, 1194.3568115234375, 284.0183410644531])
        box = np.array(box)
        masks, scores, logits = mask_predictor.predict(
            box=box,
            multimask_output=True
        )

        mask = masks[0]
        all_masks.append(mask)
    combine_mask = combine_masks(all_masks, image_rgb.shape)
    #fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    # use original image size
    fig, ax = plt.subplots(1, 1, figsize=(image_rgb.shape[1] / 100, image_rgb.shape[0] / 100))
    show_mask_on_image(image_rgb, combine_mask, ax)
    plt.axis('off')
    # plt save to /local2/wu1827_/robot/558rl/sam/results1138.jpg
    plt.savefig('/local2/wu1827_/robot/558rl/sam/results1138_.jpg')

if __name__ == "__main__":
    
    # get_bounding_box()
    sam()
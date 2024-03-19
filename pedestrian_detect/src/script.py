import cv2
import os
# from ultralytics import FastSAM
# from ultralytics.models.fastsam import FastSAMPrompt


# import imutils

def video_to_frame():
    # Load the video
    video_path = '/local2/wu1827_/robot/558rl/ahg2library.mp4'
    vidcap = cv2.VideoCapture(video_path)

    # Create a directory to store the frames
    frames_directory = "/local2/wu1827_/robot/558rl/video_frames"
    if not os.path.exists(frames_directory):
        os.makedirs(frames_directory)

    # Read through the video frame by frame
    success, image = vidcap.read()
    count = 0
    while success:
        # Save frame as JPEG file
        cv2.imwrite(f"{frames_directory}/frame{count:04d}.jpg", image)     
        success, image = vidcap.read()
        print('Read a new frame: ', success)
        count += 1
    

    print(f"Frames are extracted to {frames_directory}.")


def segment_by_prompt():

    # img_path =
    # use fast SAM model
    # Define an inference source
    source =  "/local2/wu1827_/robot/558rl/video_frames/frame0002.jpg"

    # Create a FastSAM model
    model = FastSAM('FastSAM-s.pt')  # or FastSAM-x.pt

    # Run inference on an image
    everything_results = model(source, device='gpu', retina_masks=True, imgsz=1024, conf=0.4, iou=0.9)

    # Prepare a Prompt Process object
    prompt_process = FastSAMPrompt(source, everything_results, device='gpu')

    # Everything prompt
    ann = prompt_process.everything_prompt()

    # Bbox default shape [0,0,0,0] -> [x1,y1,x2,y2]
    ann = prompt_process.box_prompt(bbox=[200, 200, 300, 300])

    # Text prompt
    ann = prompt_process.text_prompt(text='people')

    # Point prompt
    # points default [[0,0]] [[x1,y1],[x2,y2]]
    # point_label default [0] [1,0] 0:background, 1:foreground
    ann = prompt_process.point_prompt(points=[[200, 200]], pointlabel=[1])
    prompt_process.plot(annotations=ann, output='./')


'''

# Text prompt
python Inference.py --model_path /local2/wu1827_/robot/FastSAM/models/FastSAM-x.pt --img_path /local2/wu1827_/robot/558rl/video_frames/frame0010.jpg  --text_prompt "people"
'''
   
def try_opencv():

    
    # Initializing the HOG person
    # detector
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    
    # Reading the Image
    image = cv2.imread('/local2/wu1827_/robot/558rl/video_frames/frame0010.jpg')
    
    # Resizing the Image
    image = imutils.resize(image,
                        width=min(400, image.shape[1]))
    
    # Detecting all the regions in the 
    # Image that has a pedestrians inside it
    (regions, _) = hog.detectMultiScale(image, 
                                        winStride=(4, 4),
                                        padding=(4, 4),
                                        scale=1.05)
    
    # Drawing the regions in the Image
    for (x, y, w, h) in regions:
        cv2.rectangle(image, (x, y), 
                    (x + w, y + h), 
                    (0, 0, 255), 2)
    # save to /local2/wu1827_/robot/558rl/output.jpg
    cv2.imwrite('/local2/wu1827_/robot/558rl/output.jpg', image)

    # # Showing the output Image
    # cv2.imshow("Image", image)
    # cv2.waitKey(0)
    
    # cv2.destroyAllWindows()
def frames_to_video():

    img=[]
    for i in range(0,1572):
        id = str(i).zfill(4)
        img_path =f"/local2/wu1827_/robot/558rl/test/test{id}.jpg"
        assert os.path.exists(img_path)
        img.append(cv2.imread(img_path))

    height,width,layers=img[1].shape
    print("@@@@@@@@@here")
    # video=cv2.VideoWriter('/local2/wu1827_/robot/558rl/video.avi',-1,1,(width,height))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter("output.mp4", fourcc, 20.0, (width, height))
    print("@@@@@@@@@here!!!!!!")
    for j in range(0,1572):
        video.write(img[j])

    cv2.destroyAllWindows()
    video.release()
if __name__ == "__main__":
    
    video_to_frame()
    frames_to_video()
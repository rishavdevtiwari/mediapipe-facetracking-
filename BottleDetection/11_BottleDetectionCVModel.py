from ultralytics import YOLO

def main():
    # loading baseline ultra-lightweight nano model
    model = YOLO("yolov8n.pt")

    print("Starting your custom bottle training script...")
    
    # training the model on custom dataset setup
    model.train(
        data="BottleDetection/dataset/data.yaml",  # pointing to  configuration file
        epochs=25,                 # cycles through the images
        imgsz=640,                 # standard image resolution layout
        device="cpu" ,              # running computation processing on CPU
        project="BottleDetection/runs"
    )
    print("Done! Your smart weights are saved inside runs/detect/train/weights/best.pt")

if __name__ == "__main__":
    main()
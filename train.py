from ultralytics import YOLO

def main():
    try:
        # Load a model
        model = YOLO("yolo11n.pt")

        # Train the model
        train_results = model.train(
            data="D:/深度学习/ultralytics-main/data/data.yaml",  # path to dataset YAML
            epochs=150,  # number of training epochs
            imgsz=640,
            batch=10,# training image size
            device="0",  # device to run on, i.e. device=0 or device=0,1,2,3 or device=cpu
        )

        # Print training results
        print("Training Results:", train_results)

        # Evaluate model performance on the validation set
        metrics = model.val()

        # Print evaluation metrics
        print("Evaluation Metrics:", metrics)
        # Export the model to ONNX format

        path = model.export(format="onnx")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
